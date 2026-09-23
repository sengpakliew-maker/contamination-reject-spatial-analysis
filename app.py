"""Phase 10 application entry point."""

import flet as ft

from ui import build_application


def main(page: ft.Page) -> None:
    """Start the Phase 10 application."""
    build_application(page)


if __name__ == "__main__":
    ft.app(target=main)