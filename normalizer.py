import ast
import builtins
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union, Optional


@dataclass
class Context:
    prefix: str = "g_"
    current_index: int = 0
    entities_map: dict[str, str] = field(default_factory=dict)
    children: list[Union["Context"]] = field(default_factory=list)
    parent: "Context" = None


@dataclass
class ClassContext:
    prefix: str = "c_"
    current_index: int = 0


class VariableTransformer(ast.NodeTransformer):
    seen_modules: set[str] = set()
    context: list[Optional[Context]] = [Context(), None]

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        node.target = self.visit(node.target)
        return node

    def visit_Import(self, node: ast.Import) -> Any:
        if node.names[0].asname is not None:
            self.seen_modules.add(node.names[0].asname)
        else:
            self.seen_modules.add(node.names[0].name)
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        for alias in node.names:
            self.seen_modules.add(alias.name)
        return node

    def visit_Name(self, node: ast.Name) -> Any:

        match node.ctx:
            case ast.Call():
                return node
            case ast.Load() | ast.Store() | ast.Del():
                if node.id in builtins.__dict__ or node.id in self.seen_modules:
                    return node
                if node.id in self.context[0].entities_map:
                    node.id = self.context[0].entities_map[node.id]
                else:
                    current_name = f"v_{self.context[0].prefix}{self.context[1] if self.context[1] is not None else ''}{self.context[0].current_index}"
                    self.context[0].entities_map[node.id] = current_name
                    self.context[0].current_index += 1
                    node.id = current_name

        return node


class AstTransformer:
    transformer: VariableTransformer = VariableTransformer()

    def transform(self, code: str) -> str:
        tree = ast.parse(code)

        # remove docstrings
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                continue
            if not len(node.body):
                continue
            if not isinstance(node.body[0], ast.Expr):
                continue
            if not hasattr(node.body[0], 'value') or not isinstance(node.body[0].value, ast.Str):
                continue

            node.body = node.body[1:]

        self.transformer.visit(tree)
        return ast.unparse(tree)

    def transform_file(self, file_path: Path, output_file: Optional[Path] = None) -> str:
        with open(file_path, "r") as f:
            transformed = self.transform(f.read())
        if output_file is not None:
            with open(output_file, "w") as f:
                f.write(transformed)
        return transformed


if __name__ == "__main__":
    transformer = AstTransformer()
    transformer.transform_file(Path(".py"))
    transformer.transform_file(Path("program.py"), Path("main_transformed.py"))
