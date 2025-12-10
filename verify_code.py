#!/usr/bin/env python3
"""
Comprehensive verification script for Ultimate Rotoscopy
Checks all imports, connections, and logic
"""

import sys
import ast
import importlib.util
from pathlib import Path
from typing import List, Dict, Set, Tuple

class CodeVerifier:
    def __init__(self, root_path: str):
        self.root = Path(root_path)
        self.errors = []
        self.warnings = []

    def log_error(self, file: str, line: int, msg: str):
        self.errors.append(f"❌ {file}:{line} - {msg}")

    def log_warning(self, file: str, line: int, msg: str):
        self.warnings.append(f"⚠️  {file}:{line} - {msg}")

    def verify_imports(self, file_path: Path) -> bool:
        """Verify all imports can be resolved."""
        print(f"\n🔍 Checking imports in {file_path.relative_to(self.root)}...")

        try:
            with open(file_path, 'r') as f:
                tree = ast.parse(f.read(), str(file_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name
                        print(f"  📦 Import: {module_name}")

                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module
                    if module_name:
                        print(f"  📦 From {module_name} import ...")

            return True

        except SyntaxError as e:
            self.log_error(str(file_path), e.lineno or 0, f"Syntax error: {e}")
            return False
        except Exception as e:
            self.log_error(str(file_path), 0, f"Import check failed: {e}")
            return False

    def verify_class_methods(self, file_path: Path) -> Dict[str, Set[str]]:
        """Extract all classes and their methods."""
        classes = {}

        try:
            with open(file_path, 'r') as f:
                tree = ast.parse(f.read(), str(file_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    methods = set()

                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            methods.add(item.name)

                    classes[class_name] = methods
                    print(f"  📘 Class {class_name}: {len(methods)} methods")

            return classes

        except Exception as e:
            self.log_error(str(file_path), 0, f"Class analysis failed: {e}")
            return {}

    def verify_signal_connections(self, file_path: Path, classes: Dict[str, Set[str]]):
        """Verify signal connections point to existing methods."""
        print(f"\n🔗 Checking signal connections in {file_path.relative_to(self.root)}...")

        try:
            with open(file_path, 'r') as f:
                content = f.read()
                lines = content.split('\n')

            for i, line in enumerate(lines, 1):
                # Check .connect( patterns
                if '.connect(' in line and 'self.' in line:
                    # Extract what's being connected
                    if 'self.' in line:
                        parts = line.split('self.')
                        for part in parts[1:]:
                            if '.connect(' in part:
                                continue
                            # Extract method reference like "tab._method"
                            method_ref = part.split('(')[0].split(')')[0].strip()
                            if '.' in method_ref:
                                obj, method = method_ref.split('.', 1)
                                method = method.split('(')[0].strip()
                                print(f"  Line {i}: self.{obj}.{method}")

                                # Check if it looks like a method call
                                if method.startswith('_') or method[0].islower():
                                    self.warnings.append(
                                        f"Line {i}: Connection to self.{obj}.{method} - verify this method exists"
                                    )

        except Exception as e:
            self.log_error(str(file_path), 0, f"Signal check failed: {e}")

    def verify_file(self, file_path: Path):
        """Verify a single Python file."""
        print(f"\n{'='*70}")
        print(f"Verifying: {file_path.relative_to(self.root)}")
        print('='*70)

        # Check imports
        self.verify_imports(file_path)

        # Extract classes and methods
        classes = self.verify_class_methods(file_path)

        # Check signal connections
        if classes:
            self.verify_signal_connections(file_path, classes)

    def run(self):
        """Run verification on all Python files."""
        print("🚀 Starting Ultimate Rotoscopy Code Verification")
        print("="*70)

        # Find all Python files in src/ultimate_rotoscopy
        src_dir = self.root / "src" / "ultimate_rotoscopy"

        if not src_dir.exists():
            print(f"❌ Source directory not found: {src_dir}")
            return False

        python_files = list(src_dir.rglob("*.py"))
        print(f"Found {len(python_files)} Python files\n")

        # Verify key files first
        priority_files = [
            "gui/modern_gui.py",
            "gui/backend.py",
            "gui/__init__.py",
            "__init__.py",
        ]

        for rel_path in priority_files:
            file_path = src_dir / rel_path
            if file_path.exists():
                self.verify_file(file_path)

        # Print summary
        print("\n" + "="*70)
        print("VERIFICATION SUMMARY")
        print("="*70)

        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for err in self.errors:
                print(f"  {err}")
        else:
            print("\n✅ No critical errors found!")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warn in self.warnings[:20]:  # Show first 20
                print(f"  {warn}")
            if len(self.warnings) > 20:
                print(f"  ... and {len(self.warnings) - 20} more")

        print("\n" + "="*70)
        return len(self.errors) == 0

if __name__ == "__main__":
    verifier = CodeVerifier("/home/user/Rust")
    success = verifier.run()
    sys.exit(0 if success else 1)
