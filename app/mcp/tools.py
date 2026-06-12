import io
import sys
from contextlib import redirect_stdout

from fastmcp import FastMCP


mcp = FastMCP("python-tools")


@mcp.tool()
def execute_python(
    code: str,
    stdin: str = ""
) -> str:
    """
    Python 코드를 실행하고 stdout 결과를 반환한다.
    """

    buffer = io.StringIO()

    old_stdin = sys.stdin

    try:
        sys.stdin = io.StringIO(stdin)

        with redirect_stdout(buffer):
            exec(code, {})

        return buffer.getvalue()

    except Exception as e:
        return f"ERROR: {e}"

    finally:
        sys.stdin = old_stdin