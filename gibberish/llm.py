"""
LLM integration for intelligent sync planning.

This module provides integration with Ollama for LLM-powered features
including intelligent file prioritization and conflict resolution suggestions.
"""

from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import json
import time
import logging

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with local Ollama LLM instance"""

    def __init__(
        self,
        model: str = "llama3.2:latest",
        timeout: float = 15.0,
        host: str = "http://localhost:11434"
    ):
        """
        Initialize the Ollama client

        Args:
            model: Ollama model name (default: llama3.2:latest)
            timeout: Request timeout in seconds (default: 15.0)
            host: Ollama server host URL (default: http://localhost:11434)
        """
        self.model = model
        self.timeout = timeout
        self.host = host
        self._available = None  # None = not checked, True/False = checked status
        self._last_check_time = 0
        self._check_interval = 60  # Re-check availability every 60 seconds

        if not OLLAMA_AVAILABLE:
            logger.warning("ollama package not installed. LLM features will be unavailable.")
            self._available = False

    def check_availability(self) -> bool:
        """
        Check if Ollama is available and responsive

        This method caches the availability status for a short time to avoid
        repeated connection attempts.

        Returns:
            True if Ollama is reachable and has the specified model
        """
        current_time = time.time()

        # Use cached result if check was recent
        if self._available is not None and (current_time - self._last_check_time) < self._check_interval:
            return self._available

        if not OLLAMA_AVAILABLE:
            self._available = False
            return False

        try:
            # Try to list available models
            client = ollama.Client(host=self.host)
            response = client.list()

            # Check if our model is available
            available_models = [model['name'] for model in response.get('models', [])]

            # Model names might have version tags, so check for partial matches
            model_found = any(
                self.model in model_name or model_name.startswith(self.model.split(':')[0])
                for model_name in available_models
            )

            if not model_found:
                logger.warning(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Available models: {available_models}"
                )
                self._available = False
            else:
                self._available = True

            self._last_check_time = current_time
            return self._available

        except Exception as e:
            logger.debug(f"Ollama availability check failed: {e}")
            self._available = False
            self._last_check_time = current_time
            return False

    def _call_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """
        Make a call to Ollama with timeout handling

        Args:
            prompt: User prompt to send
            system_prompt: Optional system prompt for context

        Returns:
            Response text from Ollama, or None if failed/timeout
        """
        if not self.check_availability():
            logger.debug("Ollama not available, skipping LLM call")
            return None

        try:
            client = ollama.Client(host=self.host)

            messages = []
            if system_prompt:
                messages.append({
                    'role': 'system',
                    'content': system_prompt
                })
            messages.append({
                'role': 'user',
                'content': prompt
            })

            start_time = time.time()

            # Use ollama.chat for conversation-style interaction
            response = client.chat(
                model=self.model,
                messages=messages,
                options={
                    'temperature': 0.3,  # Lower temperature for more consistent results
                    'num_predict': 1024,  # Limit response length
                }
            )

            elapsed = time.time() - start_time

            # Check if we exceeded timeout
            if elapsed > self.timeout:
                logger.warning(f"Ollama response took {elapsed:.1f}s (timeout: {self.timeout}s)")
                return None

            return response['message']['content']

        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return None

    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the current model

        Returns:
            Dictionary with model information, or None if unavailable
        """
        if not self.check_availability():
            return None

        try:
            client = ollama.Client(host=self.host)
            response = client.show(self.model)
            return response
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return None


class PromptTemplates:
    """Collection of prompt templates for LLM-powered sync operations"""

    SYSTEM_SYNC_PLANNER = """You are an expert file synchronization assistant. Your role is to analyze file changes and provide intelligent recommendations for sync operations.

You should:
- Prioritize configuration files and dependencies first
- Consider file sizes and transmission efficiency
- Detect potential conflicts when multiple files modify related code
- Assess risks of applying changes
- Provide clear, actionable recommendations

Be concise and output structured data when requested."""

    PRIORITIZE_FILES = """Analyze the following list of files that need to be synchronized and recommend an optimal transmission order.

Files to sync:
{file_list}

Consider:
1. Configuration files (.json, .yaml, .toml, .env, etc.) should go first
2. Dependency files (package.json, requirements.txt, Cargo.toml, etc.) should be early
3. Smaller files should generally precede larger ones
4. Source files that other files depend on should go before dependents

Respond with a JSON array of file paths in the recommended order, with a brief explanation.
Format:
{{
  "prioritized_files": ["path1", "path2", ...],
  "reasoning": "brief explanation"
}}"""

    DETECT_CONFLICTS = """Analyze the following file changes and detect potential conflicts or issues.

Changes:
{changes}

Look for:
1. Multiple files modifying the same module/class/function
2. Changes to related configuration that might be incompatible
3. Dependency updates that could break other code
4. Binary file conflicts
5. Files being both modified and deleted

Respond with a JSON object listing any detected conflicts:
{{
  "conflicts": [
    {{
      "type": "conflict_type",
      "files": ["file1", "file2"],
      "description": "what the conflict is",
      "severity": "low|medium|high"
    }}
  ],
  "has_conflicts": true/false
}}"""

    SUGGEST_RESOLUTION = """You are helping resolve a file synchronization conflict.

Conflict description:
{conflict_description}

Provide a recommendation for how to resolve this conflict. Be specific and practical.

Respond with:
{{
  "resolution_strategy": "strategy description",
  "steps": ["step 1", "step 2", ...],
  "warnings": ["any warnings or caveats"]
}}"""

    GENERATE_SYNC_PLAN = """Create a comprehensive synchronization plan for the following changes.

Changes detected:
{changes_summary}

File statistics:
- Total files: {file_count}
- Total size: {total_size}
- Estimated transfer size (with compression): {compressed_size}

Generate a sync plan that includes:
1. Optimal file transmission order
2. Estimated completion time
3. Risk assessment
4. Any recommended pre-sync or post-sync actions

Respond with:
{{
  "transmission_order": ["file1", "file2", ...],
  "estimated_time_seconds": estimated_seconds,
  "risk_level": "low|medium|high",
  "risk_factors": ["factor 1", "factor 2", ...],
  "pre_sync_actions": ["action 1", ...],
  "post_sync_actions": ["action 1", ...],
  "summary": "brief plan summary"
}}"""


class LLMClient(OllamaClient):
    """
    Extended client with high-level LLM operations for sync planning.

    This class inherits from OllamaClient and adds domain-specific
    methods for file synchronization tasks.
    """

    def __init__(
        self,
        model: str = "llama3.2:latest",
        timeout: float = 15.0,
        host: str = "http://localhost:11434"
    ):
        """
        Initialize the LLM client

        Args:
            model: Ollama model name
            timeout: Request timeout in seconds
            host: Ollama server host URL
        """
        super().__init__(model=model, timeout=timeout, host=host)
        self.prompts = PromptTemplates()

    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """
        Parse JSON from LLM response, handling markdown code blocks

        Args:
            response: LLM response text

        Returns:
            Parsed JSON dict or None if parsing fails
        """
        if not response:
            return None

        try:
            # Try direct JSON parse first
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass

            # Try to find JSON object directly
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass

            logger.warning(f"Failed to parse JSON from LLM response: {response[:200]}")
            return None

    def prioritize_files(
        self,
        file_list: List[Path],
        file_sizes: Optional[Dict[Path, int]] = None
    ) -> Tuple[List[Path], Optional[str]]:
        """
        Use LLM to prioritize file sync order

        Args:
            file_list: List of files to prioritize
            file_sizes: Optional mapping of file paths to sizes in bytes

        Returns:
            Tuple of (prioritized file list, reasoning string)
            Falls back to original list if LLM unavailable
        """
        if not file_list:
            return [], None

        # Format file list with sizes if available
        file_descriptions = []
        for file_path in file_list:
            if file_sizes and file_path in file_sizes:
                size = file_sizes[file_path]
                file_descriptions.append(f"{file_path} ({size} bytes)")
            else:
                file_descriptions.append(str(file_path))

        file_list_str = "\n".join(f"- {desc}" for desc in file_descriptions)

        prompt = self.prompts.PRIORITIZE_FILES.format(file_list=file_list_str)
        response = self._call_ollama(prompt, system_prompt=self.prompts.SYSTEM_SYNC_PLANNER)

        if not response:
            logger.debug("LLM unavailable for file prioritization, using original order")
            return file_list, None

        # Parse response
        parsed = self._parse_json_response(response)
        if parsed and 'prioritized_files' in parsed:
            prioritized_paths = [Path(p) for p in parsed['prioritized_files']]
            reasoning = parsed.get('reasoning', '')
            return prioritized_paths, reasoning

        logger.warning("Failed to parse LLM prioritization response")
        return file_list, None

    def detect_conflicts(self, changes: List[Dict]) -> Tuple[List[Dict], bool]:
        """
        Use LLM to detect potential conflicts

        Args:
            changes: List of change descriptions with keys:
                    - 'path': file path
                    - 'type': change type (add/modify/delete)
                    - 'size': file size
                    - Optional: 'diff_summary': summary of changes

        Returns:
            Tuple of (list of conflict dicts, has_conflicts bool)
            Returns ([], False) if LLM unavailable
        """
        if not changes:
            return [], False

        # Format changes for LLM
        changes_str = json.dumps(changes, indent=2, default=str)

        prompt = self.prompts.DETECT_CONFLICTS.format(changes=changes_str)
        response = self._call_ollama(prompt, system_prompt=self.prompts.SYSTEM_SYNC_PLANNER)

        if not response:
            logger.debug("LLM unavailable for conflict detection")
            return [], False

        # Parse response
        parsed = self._parse_json_response(response)
        if parsed and 'conflicts' in parsed:
            conflicts = parsed['conflicts']
            has_conflicts = parsed.get('has_conflicts', len(conflicts) > 0)
            return conflicts, has_conflicts

        logger.warning("Failed to parse LLM conflict detection response")
        return [], False

    def suggest_resolution(self, conflict_description: str) -> Optional[Dict]:
        """
        Get LLM suggestion for conflict resolution

        Args:
            conflict_description: Description of the conflict

        Returns:
            Dictionary with resolution strategy and steps, or None if unavailable
        """
        if not conflict_description:
            return None

        prompt = self.prompts.SUGGEST_RESOLUTION.format(
            conflict_description=conflict_description
        )
        response = self._call_ollama(prompt, system_prompt=self.prompts.SYSTEM_SYNC_PLANNER)

        if not response:
            return None

        # Parse response
        parsed = self._parse_json_response(response)
        if parsed and 'resolution_strategy' in parsed:
            return parsed

        logger.warning("Failed to parse LLM resolution suggestion")
        return None

    def generate_sync_plan(
        self,
        changes: List[Dict],
        total_size: int,
        compressed_size: int
    ) -> Dict:
        """
        Generate intelligent sync plan using LLM

        Args:
            changes: List of detected changes
            total_size: Total size of changes in bytes
            compressed_size: Estimated compressed transfer size

        Returns:
            Sync plan dict with keys:
            - 'transmission_order': list of file paths
            - 'estimated_time_seconds': estimated time
            - 'risk_level': 'low', 'medium', or 'high'
            - 'risk_factors': list of risk descriptions
            - 'pre_sync_actions': list of recommended actions before sync
            - 'post_sync_actions': list of recommended actions after sync
            - 'summary': brief plan summary
            - 'llm_generated': bool indicating if plan was LLM-generated
        """
        if not changes:
            return {
                'transmission_order': [],
                'estimated_time_seconds': 0,
                'risk_level': 'low',
                'risk_factors': [],
                'pre_sync_actions': [],
                'post_sync_actions': [],
                'summary': 'No changes to sync',
                'llm_generated': False
            }

        # Format changes summary
        changes_summary = json.dumps(changes, indent=2, default=str)

        # Format file sizes
        def format_size(size_bytes: int) -> str:
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size_bytes < 1024.0:
                    return f"{size_bytes:.1f} {unit}"
                size_bytes /= 1024.0
            return f"{size_bytes:.1f} TB"

        prompt = self.prompts.GENERATE_SYNC_PLAN.format(
            changes_summary=changes_summary,
            file_count=len(changes),
            total_size=format_size(total_size),
            compressed_size=format_size(compressed_size)
        )

        response = self._call_ollama(prompt, system_prompt=self.prompts.SYSTEM_SYNC_PLANNER)

        if not response:
            logger.debug("LLM unavailable for sync plan generation")
            # Return basic plan with just file paths in original order
            return {
                'transmission_order': [c['path'] for c in changes],
                'estimated_time_seconds': max(10, compressed_size // 1024),  # Rough estimate
                'risk_level': 'medium',
                'risk_factors': ['LLM analysis unavailable'],
                'pre_sync_actions': [],
                'post_sync_actions': [],
                'summary': f'Basic sync plan for {len(changes)} files',
                'llm_generated': False
            }

        # Parse response
        parsed = self._parse_json_response(response)
        if parsed and 'transmission_order' in parsed:
            parsed['llm_generated'] = True
            return parsed

        logger.warning("Failed to parse LLM sync plan response")
        # Return fallback plan
        return {
            'transmission_order': [c['path'] for c in changes],
            'estimated_time_seconds': max(10, compressed_size // 1024),
            'risk_level': 'medium',
            'risk_factors': ['Failed to parse LLM response'],
            'pre_sync_actions': [],
            'post_sync_actions': [],
            'summary': f'Fallback sync plan for {len(changes)} files',
            'llm_generated': False
        }


class RuleBasedPlanner:
    """
    Rule-based file prioritization and sync planning fallback.

    This class provides deterministic, fast file prioritization when
    LLM is unavailable or times out. Uses heuristic rules to order files.
    """

    # File extension categories (priority order)
    CONFIG_EXTENSIONS = {
        '.json', '.yaml', '.yml', '.toml', '.ini', '.conf', '.config',
        '.env', '.properties', '.xml'
    }

    DEPENDENCY_FILES = {
        'package.json', 'requirements.txt', 'Pipfile', 'pyproject.toml',
        'Cargo.toml', 'go.mod', 'pom.xml', 'build.gradle', 'composer.json',
        'Gemfile', 'mix.exs'
    }

    CODE_EXTENSIONS = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.rs', '.go', '.java',
        '.c', '.cpp', '.h', '.hpp', '.cs', '.rb', '.php', '.swift'
    }

    DOC_EXTENSIONS = {
        '.md', '.txt', '.rst', '.adoc', '.html', '.htm'
    }

    BINARY_EXTENSIONS = {
        '.png', '.jpg', '.jpeg', '.gif', '.ico', '.pdf', '.zip', '.tar',
        '.gz', '.exe', '.dll', '.so', '.dylib', '.bin', '.dat'
    }

    def __init__(self):
        """Initialize the rule-based planner"""
        pass

    def _get_file_priority(self, file_path: Path, file_size: int = 0) -> Tuple[int, int]:
        """
        Calculate priority for a file based on rules

        Args:
            file_path: Path to file
            file_size: File size in bytes

        Returns:
            Tuple of (category_priority, size_priority)
            Lower numbers = higher priority
        """
        filename = file_path.name.lower()
        extension = file_path.suffix.lower()

        # Category priority (0 = highest)
        if filename in self.DEPENDENCY_FILES:
            category = 0  # Dependency files first
        elif extension in self.CONFIG_EXTENSIONS:
            category = 1  # Config files second
        elif extension in self.CODE_EXTENSIONS:
            category = 2  # Source code third
        elif extension in self.DOC_EXTENSIONS:
            category = 3  # Documentation fourth
        elif extension in self.BINARY_EXTENSIONS:
            category = 4  # Binary files last
        else:
            category = 2  # Unknown files with code

        # Size priority - smaller files first within each category
        # Use log scale to avoid huge numbers
        size_priority = file_size if file_size > 0 else 0

        return (category, size_priority)

    def prioritize_files(
        self,
        file_list: List[Path],
        file_sizes: Optional[Dict[Path, int]] = None
    ) -> List[Path]:
        """
        Prioritize files using rule-based logic

        Args:
            file_list: List of files to prioritize
            file_sizes: Optional mapping of file paths to sizes

        Returns:
            Prioritized list of files (configs first, small before large)
        """
        if not file_list:
            return []

        # Create list of (file, priority) tuples
        file_priorities = []
        for file_path in file_list:
            size = file_sizes.get(file_path, 0) if file_sizes else 0
            priority = self._get_file_priority(file_path, size)
            file_priorities.append((file_path, priority))

        # Sort by priority (category first, then size)
        file_priorities.sort(key=lambda x: x[1])

        # Return just the sorted file paths
        return [fp[0] for fp in file_priorities]

    def analyze_dependencies(self, file_list: List[Path]) -> Dict[str, List[str]]:
        """
        Analyze simple file dependencies based on imports/includes

        This is a basic heuristic analyzer - for production, use a proper
        dependency graph tool.

        Args:
            file_list: List of files to analyze

        Returns:
            Dictionary mapping file paths to list of files they might depend on
        """
        dependencies = {}

        for file_path in file_list:
            deps = []
            extension = file_path.suffix.lower()

            # Simple heuristic: if there's a config file, code files might depend on it
            if extension in self.CODE_EXTENSIONS:
                for other_file in file_list:
                    if other_file.suffix.lower() in self.CONFIG_EXTENSIONS:
                        deps.append(str(other_file))

            dependencies[str(file_path)] = deps

        return dependencies

    def detect_simple_conflicts(self, changes: List[Dict]) -> List[Dict]:
        """
        Detect simple conflicts using rule-based heuristics

        Args:
            changes: List of change dicts with 'path', 'type', etc.

        Returns:
            List of detected conflict dicts
        """
        conflicts = []

        # Group changes by directory
        dir_changes = {}
        for change in changes:
            path = Path(change.get('path', ''))
            dir_path = str(path.parent)

            if dir_path not in dir_changes:
                dir_changes[dir_path] = []
            dir_changes[dir_path].append(change)

        # Check for multiple changes in same directory
        for dir_path, dir_change_list in dir_changes.items():
            if len(dir_change_list) > 5:
                conflicts.append({
                    'type': 'high_churn_directory',
                    'files': [c['path'] for c in dir_change_list],
                    'description': f'{len(dir_change_list)} files changed in {dir_path}',
                    'severity': 'low'
                })

        # Check for config file changes
        config_changes = [
            c for c in changes
            if Path(c.get('path', '')).suffix.lower() in self.CONFIG_EXTENSIONS
        ]

        if config_changes:
            conflicts.append({
                'type': 'config_modification',
                'files': [c['path'] for c in config_changes],
                'description': 'Configuration files modified - may affect other files',
                'severity': 'medium'
            })

        # Check for dependency file changes
        dep_changes = [
            c for c in changes
            if Path(c.get('path', '')).name.lower() in self.DEPENDENCY_FILES
        ]

        if dep_changes:
            conflicts.append({
                'type': 'dependency_modification',
                'files': [c['path'] for c in dep_changes],
                'description': 'Dependency files changed - may require reinstall',
                'severity': 'high'
            })

        return conflicts

    def generate_basic_sync_plan(
        self,
        changes: List[Dict],
        total_size: int,
        compressed_size: int
    ) -> Dict:
        """
        Generate a basic sync plan using rules

        Args:
            changes: List of change dicts
            total_size: Total size in bytes
            compressed_size: Compressed size estimate

        Returns:
            Sync plan dict
        """
        if not changes:
            return {
                'transmission_order': [],
                'estimated_time_seconds': 0,
                'risk_level': 'low',
                'risk_factors': [],
                'pre_sync_actions': [],
                'post_sync_actions': [],
                'summary': 'No changes to sync',
                'llm_generated': False
            }

        # Build file list and sizes
        file_list = [Path(c['path']) for c in changes]
        file_sizes = {Path(c['path']): c.get('size', 0) for c in changes}

        # Prioritize files
        ordered_files = self.prioritize_files(file_list, file_sizes)

        # Detect conflicts
        conflicts = self.detect_simple_conflicts(changes)

        # Estimate time (rough: 1KB/sec for acoustic transmission)
        estimated_time = max(10, compressed_size // 1024)

        # Determine risk level
        risk_factors = []
        if any(c['severity'] == 'high' for c in conflicts):
            risk_level = 'high'
            risk_factors.append('High-severity changes detected')
        elif len(conflicts) > 0:
            risk_level = 'medium'
            risk_factors.append(f'{len(conflicts)} potential issues detected')
        else:
            risk_level = 'low'

        # Generate actions
        pre_sync_actions = []
        post_sync_actions = []

        if any(c['type'] == 'dependency_modification' for c in conflicts):
            post_sync_actions.append('Reinstall dependencies after sync')

        if any(c['type'] == 'config_modification' for c in conflicts):
            post_sync_actions.append('Verify configuration settings')

        return {
            'transmission_order': [str(f) for f in ordered_files],
            'estimated_time_seconds': estimated_time,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'pre_sync_actions': pre_sync_actions,
            'post_sync_actions': post_sync_actions,
            'summary': f'Rule-based sync plan for {len(changes)} files',
            'llm_generated': False
        }


class SyncPlanner:
    """
    Unified sync planner that combines LLM and rule-based approaches.

    This class provides intelligent sync planning with automatic fallback:
    1. Try LLM-based planning first (if available and within timeout)
    2. Fall back to rule-based planning if LLM unavailable or slow
    3. Always provide a valid sync plan
    """

    def __init__(
        self,
        llm_model: str = "llama3.2:latest",
        llm_timeout: float = 15.0,
        llm_host: str = "http://localhost:11434",
        enable_llm: bool = True
    ):
        """
        Initialize the sync planner

        Args:
            llm_model: Ollama model name
            llm_timeout: LLM request timeout in seconds (default: 15.0)
            llm_host: Ollama server host URL
            enable_llm: Whether to enable LLM features (default: True)
        """
        self.enable_llm = enable_llm and OLLAMA_AVAILABLE
        self.llm_timeout = llm_timeout

        if self.enable_llm:
            self.llm_client = LLMClient(
                model=llm_model,
                timeout=llm_timeout,
                host=llm_host
            )
        else:
            self.llm_client = None

        self.rule_planner = RuleBasedPlanner()

    def prioritize_files(
        self,
        file_list: List[Path],
        file_sizes: Optional[Dict[Path, int]] = None,
        prefer_llm: bool = True
    ) -> Tuple[List[Path], str]:
        """
        Prioritize files for sync with automatic fallback

        Args:
            file_list: List of files to prioritize
            file_sizes: Optional mapping of file paths to sizes
            prefer_llm: Try LLM first if True, use rules if False

        Returns:
            Tuple of (prioritized file list, method used: 'llm' or 'rules')
        """
        if not file_list:
            return [], 'none'

        # Try LLM if enabled and preferred
        if self.enable_llm and prefer_llm and self.llm_client:
            try:
                prioritized, reasoning = self.llm_client.prioritize_files(
                    file_list, file_sizes
                )
                if prioritized:  # LLM succeeded
                    logger.info(f"LLM prioritization: {reasoning}")
                    return prioritized, 'llm'
            except Exception as e:
                logger.warning(f"LLM prioritization failed: {e}")

        # Fall back to rule-based
        logger.debug("Using rule-based file prioritization")
        prioritized = self.rule_planner.prioritize_files(file_list, file_sizes)
        return prioritized, 'rules'

    def detect_conflicts(
        self,
        changes: List[Dict],
        prefer_llm: bool = True
    ) -> Tuple[List[Dict], bool, str]:
        """
        Detect conflicts with automatic fallback

        Args:
            changes: List of change dicts
            prefer_llm: Try LLM first if True

        Returns:
            Tuple of (conflicts list, has_conflicts bool, method used)
        """
        if not changes:
            return [], False, 'none'

        # Try LLM if enabled and preferred
        if self.enable_llm and prefer_llm and self.llm_client:
            try:
                conflicts, has_conflicts = self.llm_client.detect_conflicts(changes)
                if conflicts is not None:  # LLM returned something
                    return conflicts, has_conflicts, 'llm'
            except Exception as e:
                logger.warning(f"LLM conflict detection failed: {e}")

        # Fall back to rule-based
        logger.debug("Using rule-based conflict detection")
        conflicts = self.rule_planner.detect_simple_conflicts(changes)
        has_conflicts = len(conflicts) > 0
        return conflicts, has_conflicts, 'rules'

    def generate_sync_plan(
        self,
        changes: List[Dict],
        total_size: int,
        compressed_size: int,
        prefer_llm: bool = True
    ) -> Dict:
        """
        Generate comprehensive sync plan with automatic fallback

        Args:
            changes: List of change dicts
            total_size: Total size in bytes
            compressed_size: Estimated compressed size
            prefer_llm: Try LLM first if True

        Returns:
            Sync plan dict with 'llm_generated' flag and 'planner_used' field
        """
        if not changes:
            return {
                'transmission_order': [],
                'estimated_time_seconds': 0,
                'risk_level': 'low',
                'risk_factors': [],
                'pre_sync_actions': [],
                'post_sync_actions': [],
                'summary': 'No changes to sync',
                'llm_generated': False,
                'planner_used': 'none'
            }

        # Try LLM if enabled and preferred
        if self.enable_llm and prefer_llm and self.llm_client:
            try:
                plan = self.llm_client.generate_sync_plan(
                    changes, total_size, compressed_size
                )
                if plan.get('llm_generated', False):
                    logger.info("Generated LLM-powered sync plan")
                    plan['planner_used'] = 'llm'
                    return plan
            except Exception as e:
                logger.warning(f"LLM sync plan generation failed: {e}")

        # Fall back to rule-based
        logger.debug("Using rule-based sync planning")
        plan = self.rule_planner.generate_basic_sync_plan(
            changes, total_size, compressed_size
        )
        plan['planner_used'] = 'rules'
        return plan

    def suggest_conflict_resolution(
        self,
        conflict_description: str
    ) -> Optional[Dict]:
        """
        Get conflict resolution suggestion (LLM only, no fallback)

        Args:
            conflict_description: Description of the conflict

        Returns:
            Resolution dict or None if LLM unavailable
        """
        if not self.enable_llm or not self.llm_client:
            return None

        try:
            return self.llm_client.suggest_resolution(conflict_description)
        except Exception as e:
            logger.warning(f"LLM resolution suggestion failed: {e}")
            return None

    def get_planner_status(self) -> Dict[str, Any]:
        """
        Get status information about the planner

        Returns:
            Dict with planner status information
        """
        status = {
            'llm_enabled': self.enable_llm,
            'rule_based_available': True,
            'timeout_seconds': self.llm_timeout
        }

        if self.enable_llm and self.llm_client:
            status['llm_available'] = self.llm_client.check_availability()
            status['llm_model'] = self.llm_client.model
            status['llm_host'] = self.llm_client.host
        else:
            status['llm_available'] = False
            status['llm_model'] = None
            status['llm_host'] = None

        return status
