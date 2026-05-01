# This Python code is for the GUI to interact with the ViewSonic LS740HD commands.

import hmac
import os
import secrets
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from serial.tools import list_ports
from starlette.middleware.sessions import SessionMiddleware

from app.serial_comm import ProjectorSerial
from app.controller import ProjectorController

app = FastAPI()
SESSION_SECRET_KEY = secrets.token_urlsafe(32)
SESSION_MAX_AGE_SECONDS = 60 * 60 * 24 * 30  # 30 days

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET_KEY,
    same_site="lax",
    max_age=SESSION_MAX_AGE_SECONDS,
)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
RESTART_SCRIPT = BASE_DIR / "restart.sh"
RESTART_OUTPUT_LOG = BASE_DIR / "terminal.log"

AUTH_USERNAME = "FONTAINE_DECAL"
AUTH_PASSWORD = "FONTAINE_DECAL"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _default_port() -> str:
    configured_port = os.getenv("PROJECTOR_PORT")
    if configured_port:
        return configured_port

    available_ports = [port.device for port in list_ports.comports()]
    if available_ports:
        return sorted(available_ports)[0]

    return "COM3" if os.name == "nt" else "/dev/ttyUSB0"

# Allow the UI to load even when the serial device is unavailable.
serial_iface = None
projector = None
serial_error = None
current_port = _default_port()
command_lock = Lock()
pipeline_lock = Lock()
pipeline_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="projector-pipeline")

state = {
    "power": "Off",
    "input": "HDMI 1",
    "connected": False,
    "port": current_port,
    "connection_detail": "Starting connection check...",
    "last_update": "--",
    # Progress tracking for pipeline (step index, total steps, description)
    "progress_step": 0,
    "progress_total": 0,
    "progress_description": "IDLE",
    "pipeline_running": False,
    "pipeline_status": "idle",
}

event_log = []
progress_lock = Lock()


class ConnectRequest(BaseModel):
    port: str


class LoginRequest(BaseModel):
    username: str
    password: str


class ProgressRequest(BaseModel):
    step: int
    total: int | None = None
    description: str | None = None


def _timestamp() -> str:
    return datetime.now().strftime("%I:%M:%S %p")


def _set_progress(
    *,
    step: int | None = None,
    total: int | None = None,
    description: str | None = None,
    pipeline_running: bool | None = None,
    pipeline_status: str | None = None,
) -> None:
    with progress_lock:
        if total is not None:
            state["progress_total"] = max(0, int(total))

        if step is not None:
            next_step = max(0, int(step))
            if total is None and next_step > int(state.get("progress_total", 0)):
                state["progress_total"] = next_step
            max_step = int(state.get("progress_total", 0))
            state["progress_step"] = min(next_step, max_step) if max_step > 0 else next_step

        if description is not None:
            state["progress_description"] = description

        if pipeline_running is not None:
            state["pipeline_running"] = pipeline_running

        if pipeline_status is not None:
            state["pipeline_status"] = pipeline_status

        state["last_update"] = _timestamp()


def _is_authenticated(request: Request) -> bool:
    return request.session.get("authenticated") is True


def _api_unauthorized_response() -> JSONResponse:
    return JSONResponse({"success": False, "detail": "Authentication required"}, status_code=401)


def _render_login_html(error: str = "") -> str:
    error_block = f'<div class="login-error">{error}</div>' if error else ""
    return f"""
    <!DOCTYPE html>
    <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Login - Automatic Decal Projection</title>
            <link rel="icon" type="image/x-icon" href="/static/favicon.ico">
            <link rel="icon" type="image/png" sizes="32x32" href="/static/tab-logo-transparent.png">
            <style>
                :root {{
                    --bg-top: #161719;
                    --bg-bottom: #0b0c0d;
                    --panel: rgba(31, 33, 36, 0.95);
                    --border: rgba(154, 154, 154, 0.2);
                    --text-main: #f2f2f3;
                    --text-muted: #b4b6ba;
                    --danger: #f87171;
                }}
                * {{ box-sizing: border-box; }}
                body {{
                    margin: 0;
                    min-height: 100vh;
                    display: grid;
                    place-items: center;
                    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
                    color: var(--text-main);
                    background:
                        radial-gradient(circle at 12% 10%, rgba(128, 128, 128, 0.08), transparent 28%),
                        linear-gradient(160deg, var(--bg-top), var(--bg-bottom) 58%, #080909);
                }}
                .login-card {{
                    width: min(440px, calc(100% - 32px));
                    padding: 24px;
                    border-radius: 10px;
                    border: 1px solid var(--border);
                    background: var(--panel);
                    box-shadow: 0 14px 44px rgba(0, 0, 0, 0.52);
                }}
                .login-logo-wrap {{
                    display: grid;
                    place-items: center;
                    margin-bottom: 10px;
                }}
                .login-logo {{
                    width: 156px;
                    height: auto;
                }}
                .brand {{ margin: 0; color: var(--text-muted); font-size: 0.85rem; letter-spacing: 0.08em; text-transform: uppercase; font-weight: 700; }}
                h1 {{ margin: 8px 0 0; font-size: 1.4rem; }}
                .row {{ margin-top: 14px; }}
                label {{ display: block; margin-bottom: 6px; color: var(--text-muted); font-weight: 700; font-size: 0.82rem; letter-spacing: 0.06em; text-transform: uppercase; }}
                input {{
                    width: 100%;
                    padding: 10px 12px;
                    border-radius: 8px;
                    border: 1px solid rgba(120, 120, 120, 0.35);
                    background: rgba(18, 20, 23, 0.9);
                    color: var(--text-main);
                    font-size: 0.95rem;
                }}
                .login-btn {{
                    margin-top: 16px;
                    width: 100%;
                    padding: 10px 14px;
                    border: 1px solid rgba(100, 100, 100, 0.24);
                    border-radius: 8px;
                    background: linear-gradient(135deg, #45484f, #353940);
                    color: #fff;
                    font-weight: 700;
                    font-size: 0.85rem;
                    letter-spacing: 0.02em;
                    cursor: pointer;
                }}
                .login-error {{ margin-top: 12px; color: var(--danger); font-weight: 700; }}
            </style>
        </head>
        <body>
            <main class="login-card">
                <div class="login-logo-wrap">
                    <img class="login-logo" src="/static/fontaine-logo-white.png" alt="Automatic Decal Projection logo">
                </div>
                <p class="brand">Fontaine Modification</p>
                <h1>User Login</h1>
                {error_block}
                <form id="login-form">
                    <div class="row">
                        <label for="username">Username</label>
                        <input id="username" type="text" autocomplete="username" required>
                    </div>
                    <div class="row">
                        <label for="password">Password</label>
                        <input id="password" type="password" autocomplete="current-password" required>
                    </div>
                    <button class="login-btn" type="submit">SIGN IN</button>
                </form>
            </main>

            <script>
                document.getElementById('login-form').addEventListener('submit', async (event) => {{
                    event.preventDefault();
                    const username = document.getElementById('username').value;
                    const password = document.getElementById('password').value;

                    const response = await fetch('/login', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ username, password }})
                    }});

                    if (response.ok) {{
                        localStorage.setItem('projector-theme', 'dark');
                        window.location.href = '/';
                        return;
                    }}

                    window.location.href = '/login?error=1';
                }});
            </script>
        </body>
    </html>
    """


def _record_event(message: str, success: bool) -> None:
    event_log.append({"time": _timestamp(), "message": message, "success": success})
    if len(event_log) > 80:
        del event_log[:-80]


def _humanize_serial_error(error: str | None) -> str:
    if not error:
        return "Connection OK"

    lowered = error.lower()
    if "could not open port" in lowered and "filenotfounderror" in lowered:
        return f"{current_port} not available"
    if "permissionerror" in lowered or "access is denied" in lowered:
        return f"{current_port} busy"
    if "timed out" in lowered:
        return f"{current_port} timeout"
    return f"{current_port} connection failed"


def _connection_detail() -> str:
    if state.get("connected"):
        return f"{current_port} connected"
    return _humanize_serial_error(serial_error)


def _header_connection_detail() -> str:
    if state.get("connected"):
        return "Projector connection active"
    return "Projector connection unavailable"


def _windows_process_command_line(pid: int) -> str:
    if os.name != "nt" or pid <= 0:
        return ""

    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"(Get-CimInstance Win32_Process -Filter 'ProcessId = {pid}').CommandLine",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return (result.stdout or "").strip().lower()
    except Exception:
        return ""


def _shutdown_runtime_resources() -> None:
    global serial_iface, projector

    try:
        if serial_iface is not None:
            print("Closing serial connection...")
            serial_iface.close()
    except Exception as exc:
        print(f"Error closing serial: {exc}")

    serial_iface = None
    projector = None

    try:
        pipeline_executor.shutdown(wait=False, cancel_futures=True)
    except Exception as exc:
        print(f"Error shutting down pipeline executor: {exc}")


def _launch_restart_script() -> None:
    if os.name == "nt":
        raise RuntimeError("Force restart is configured for the Linux startup script.")

    if not RESTART_SCRIPT.exists():
        raise RuntimeError(f"Restart script not found: {RESTART_SCRIPT}")

    log_handle = open(RESTART_OUTPUT_LOG, "ab", buffering=0)
    try:
        subprocess.Popen(
            ["bash", str(RESTART_SCRIPT)],
            cwd=BASE_DIR,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log_handle.close()


def _terminate_server_process() -> None:
    current_pid = os.getpid()

    if os.name == "nt":
        parent_pid = os.getppid()
        target_pid = current_pid
        parent_command = _windows_process_command_line(parent_pid)

        # In uvicorn --reload, requests are handled by a child worker.
        # Kill the reloader parent tree so the server does not restart.
        if "uvicorn" in parent_command and "app.gui:app" in parent_command:
            target_pid = parent_pid

        try:
            subprocess.run(
                ["taskkill", "/PID", str(target_pid), "/T", "/F"],
                check=False,
                capture_output=True,
                timeout=3,
            )
            return
        except Exception:
            pass

    # Fallback for non-Windows or if taskkill fails.
    if "--reload" in " ".join(sys.argv).lower():
        os._exit(0)
    os._exit(0)


def _connect_projector(port: str) -> bool:
    global serial_iface, projector, serial_error, current_port

    try:
        if serial_iface is not None:
            serial_iface.close()
    except Exception:
        pass

    serial_iface = None
    projector = None
    serial_error = None
    current_port = port
    state["port"] = port

    try:
        serial_iface = ProjectorSerial(port=port)
        projector = ProjectorController(serial_iface)
        state["connected"] = True
        state["power"] = "Off"
        state["connection_detail"] = _connection_detail()
        state["last_update"] = _timestamp()
        _record_event(f"Connected to {port}", True)
        return True
    except Exception as exc:
        serial_error = str(exc)
        state["connected"] = False
        state["power"] = "Off"
        state["connection_detail"] = _connection_detail()
        state["last_update"] = _timestamp()
        _record_event(f"Failed to connect to {port}", False)
        return False

_connect_projector(current_port)

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if _is_authenticated(request):
        return RedirectResponse(url="/", status_code=303)

    show_error = request.query_params.get("error") == "1"
    error = "Invalid username or password" if show_error else ""
    return HTMLResponse(_render_login_html(error))


@app.post("/login")
def login_submit(payload: LoginRequest, request: Request):
    if hmac.compare_digest(payload.username, AUTH_USERNAME) and hmac.compare_digest(payload.password, AUTH_PASSWORD):
        request.session["authenticated"] = True
        return {"success": True}
    return JSONResponse({"success": False, "detail": "Invalid credentials"}, status_code=401)


@app.post("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


@app.post("/shutdown")
def shutdown(request: Request, background_tasks: BackgroundTasks):
    if not _is_authenticated(request):
        return _api_unauthorized_response()

    def _do_shutdown():
        print("Shutting down application...")
        _shutdown_runtime_resources()
        _terminate_server_process()

    # Run shutdown after the response is sent so the browser gets a success payload.
    background_tasks.add_task(_do_shutdown)
    return {"status": "shutting down", "success": True}


@app.post("/restart")
def restart(request: Request, background_tasks: BackgroundTasks):
    if not _is_authenticated(request):
        return _api_unauthorized_response()

    try:
        _launch_restart_script()
    except Exception as exc:
        return JSONResponse(
            {"success": False, "detail": str(exc)},
            status_code=500,
        )

    def _do_restart():
        print("Force restarting application...")
        _shutdown_runtime_resources()
        _terminate_server_process()

    # Start the detached restart script first, then stop this process after the response.
    background_tasks.add_task(_do_restart)
    return {"status": "restarting", "success": True}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    if not _is_authenticated(request):
        return RedirectResponse(url="/login", status_code=303)

    connection_label = "Connected" if state.get("connected") else "Disconnected"
    connection_class = "status-badge connected" if state.get("connected") else "status-badge disconnected"

    return f"""
    <!DOCTYPE html>
    <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Dashboard - Automatic Decal Projection</title>
            <link rel="icon" type="image/x-icon" href="/static/favicon.ico">
            <link rel="icon" type="image/png" sizes="32x32" href="/static/tab-logo-transparent.png">
            <link rel="stylesheet" href="/static/style.css">
        </head>
        <body>
            <div class="background-orb orb-a"></div>
            <div class="background-orb orb-b"></div>
            <style>
                .profile-menu-fixed {{
                    position: fixed;
                    top: 16px;
                    right: 16px;
                    z-index: 9999;
                    display: flex;
                    align-items: center;
                }}
                .profile-trigger {{
                    position: relative;
                }}
                .profile-button {{
                    width: 40px;
                    height: 40px;
                    border-radius: 50%;
                    border: 1px solid rgba(120, 120, 120, 0.35);
                    background: rgba(18, 20, 23, 0.95);
                    color: #b4b6ba;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 0;
                    transition: all 0.2s ease;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
                }}
                .profile-button:hover {{
                    background: rgba(30, 35, 40, 0.98);
                    border-color: rgba(154, 154, 154, 0.5);
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
                }}
                .profile-dropdown {{
                    position: absolute;
                    top: 100%;
                    right: 0;
                    margin-top: 8px;
                    background: rgba(31, 33, 36, 0.98);
                    border: 1px solid rgba(154, 154, 154, 0.2);
                    border-radius: 8px;
                    overflow: hidden;
                    min-width: 140px;
                    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6);
                    display: none;
                }}
                .profile-dropdown.visible {{
                    display: block;
                }}
                .profile-option {{
                    width: 100%;
                    padding: 12px 14px;
                    background: none;
                    border: none;
                    color: #b4b6ba;
                    text-align: left;
                    cursor: pointer;
                    font-size: 0.9rem;
                    font-weight: 500;
                    transition: all 0.15s ease;
                }}
                .profile-option:hover {{
                    background: rgba(70, 100, 140, 0.25);
                    color: #f2f2f3;
                }}
                .profile-option.profile-shutdown {{
                    border-top: 1px solid rgba(154, 154, 154, 0.15);
                    color: #f87171;
                    background: rgba(248, 113, 113, 0.08);
                }}
                .profile-option.profile-shutdown:hover {{
                    background: rgba(248, 113, 113, 0.28);
                    color: #ff9999;
                }}
                body[data-theme="light"] .profile-button {{
                    background: rgba(249, 250, 252, 0.98);
                    border-color: rgba(120, 126, 136, 0.42);
                    color: #1f2329;
                    box-shadow: 0 2px 8px rgba(40, 40, 40, 0.16);
                }}
                body[data-theme="light"] .profile-button:hover {{
                    background: rgba(238, 240, 244, 0.98);
                    border-color: rgba(105, 112, 123, 0.52);
                    box-shadow: 0 4px 12px rgba(40, 40, 40, 0.22);
                }}
                body[data-theme="light"] .profile-dropdown {{
                    background: rgba(252, 252, 253, 0.99);
                    border-color: rgba(120, 126, 136, 0.25);
                }}
                body[data-theme="light"] .profile-option {{
                    color: #2b3139;
                }}
                body[data-theme="light"] .profile-option:hover {{
                    background: rgba(70, 100, 140, 0.14);
                    color: #15181d;
                }}
                body[data-theme="light"] .profile-option.profile-shutdown {{
                    color: #f87171;
                    background: rgba(248, 113, 113, 0.08);
                }}
                body[data-theme="light"] .profile-option.profile-shutdown:hover {{
                    background: rgba(248, 113, 113, 0.16);
                    color: #f87171;
                }}
                .title-wrap {{
                    display: flex;
                    align-items: center;
                    gap: 12px;
                }}
                .title-logo {{
                    width: 120px;
                    height: auto;
                    opacity: 0.95;
                    flex-shrink: 0;
                    transition: filter 0.2s ease;
                }}
                body[data-theme="light"] .title-logo {{
                    filter: brightness(0) saturate(100%);
                }}
                .watermark {{
                    position: fixed;
                    bottom: 0;
                    left: 50%;
                    transform: translateX(-50%);
                    opacity: 0.3;
                    z-index: 1;
                }}
                .watermark img {{
                    width: 200px;
                    height: auto;
                }}
                body[data-theme="light"] .watermark img {{
                    filter: invert(1);
                }}
                .save-result-btn:disabled {{
                    opacity: 0.4;
                    cursor: not-allowed;
                }}
                .save-result-btn:disabled:hover {{
                    background: no-change;
                    transform: none;
                }}
            </style>
            <main class="shell">
                <div id="profile-menu" class="profile-menu-fixed">
                    <div class="profile-trigger">
                        <button class="profile-button" onclick="toggleProfileMenu()" title="Menu">
                            <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
                                <circle cx="12" cy="8" r="4"/>
                                <path d="M12 14c-6 0-8 3-8 3v3h16v-3s-2-3-8-3z"/>
                            </svg>
                        </button>
                        <div id="profile-dropdown" class="profile-dropdown">
                            <button class="profile-option" onclick="logout()">Logout</button>
                            <button class="profile-option profile-shutdown" onclick="restartApp()">Force Restart</button>
                            <button class="profile-option profile-shutdown" onclick="shutdownApp()">Shutdown</button>
                        </div>
                    </div>
                </div>

                <section class="hero-card">
                    <div class="hero-row">
                        <div class="title-wrap">
                            <img class="title-logo" src="/static/fontaine-logo-white.png" alt="Automatic Decal Projection logo">
                            <h1>Automatic Decal Projection</h1>
                        </div>
                        <div class="hero-actions">
                            <button id="theme-toggle" class="theme-toggle" type="button" role="switch" aria-label="Toggle theme" aria-checked="false" title="Toggle theme">
                                <svg class="toggle-icon toggle-sun" viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
                                    <circle cx="12" cy="12" r="4.5" fill="none" stroke="currentColor" stroke-width="2"/>
                                    <path d="M12 2.5v2.5M12 19v2.5M21.5 12H19M5 12H2.5M18.72 5.28l-1.77 1.77M7.05 16.95l-1.77 1.77M18.72 18.72l-1.77-1.77M7.05 7.05L5.28 5.28" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                                </svg>
                                <span class="toggle-slider"></span>
                                <svg class="toggle-icon toggle-moon" viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" fill="currentColor"/></svg>
                            </button>
                            <div class="{connection_class}">{connection_label}</div>
                        </div>
                    </div>
                </section>

                <section class="panel-grid">
                    <article class="control-card accent-blue">
                        <div class="card-label">PROGRAM</div>
                        <div class="button-row">
                            <button class="action-button start-btn command-button" onclick="sendCommand('/start')">START</button>
                            <button class="action-button command-button power-off" onclick="sendCommand('/progress/reset', 'Reset pipeline progress?')">RESET</button>
                            <button id="save-result-btn" class="action-button command-button save-result-btn" onclick="saveResult()" disabled>SAVE RESULT</button>
                        </div>
                        <div class="progress-wrap" style="margin-top:12px;">
                            <div class="progress-bar" aria-hidden="true" style="display:flex;gap:6px;">
                                <div class="progress-segment" data-index="1" style="flex:1;height:12px;border-radius:6px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.04);"></div>
                                <div class="progress-segment" data-index="2" style="flex:1;height:12px;border-radius:6px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.04);"></div>
                                <div class="progress-segment" data-index="3" style="flex:1;height:12px;border-radius:6px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.04);"></div>
                                <div class="progress-segment" data-index="4" style="flex:1;height:12px;border-radius:6px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.04);"></div>
                                <div class="progress-segment" data-index="5" style="flex:1;height:12px;border-radius:6px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.04);"></div>
                                <div class="progress-segment" data-index="6" style="flex:1;height:12px;border-radius:6px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.04);"></div>
                                <div class="progress-segment" data-index="7" style="flex:1;height:12px;border-radius:6px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.04);"></div>
                            </div>
                            <div class="progress-caption" style="margin-top:8px;font-weight:700;color:var(--text-muted);display:flex;align-items:center;gap:8px;">
                                <span id="progress-check" class="ui-checkmark" style="color:var(--start-green);display:none;">✓</span>
                                <span id="progress-text">STEP (0/7)</span>
                                <span id="progress-desc" style="color:var(--start-green);font-weight:500;font-size:1rem;margin-left:8px;"></span>
                            </div>
                        </div>
                    </article>

                    <article class="control-card accent-blue">
                        <div class="card-label">PROJECTOR</div>
                        <div class="button-row">
                                <button class="action-button power-on command-button" onclick="sendCommand('/power/on')">POWER ON</button>
                                <button class="action-button power-off command-button" onclick="sendCommand('/power/off', 'Confirm POWER OFF?')">POWER OFF</button>
                            </div>
                            <div class="button-row">
                                
                                <button class="action-button command-button" onclick="sendCommand('/av/mute/off')">DISPLAY ON</button>
                                <button class="action-button command-button power-off" onclick="sendCommand('/av/mute/on')">DISPLAY OFF</button>
                        </div>
                    </article>

                    <article class="control-card accent-slate">
                        <div class="card-label">STATUS</div>
                        <div id="command-feedback" class="status-block">
                            <div id="feedback-text">
                                <span id="feedback-check" class="ui-checkmark" style="display:none;">✓</span>
                                <span id="feedback-message">Ready</span>
                            </div>
                        </div>
                        <div id="status-display" class="status-block">
                            <div><span class="status-name">Power</span><span class="status-value" id="status-power">--</span></div>
                            <div><span class="status-name">Input</span><span class="status-value" id="status-input">--</span></div>
                            <div><span class="status-name">Connection</span><span class="status-value" id="status-connection">--</span></div>
                            <div><span class="status-name">Last update</span><span class="status-value" id="status-time">--</span></div>
                        </div>
                    </article>

                    <article class="control-card accent-blue">
                        <div class="card-label">CONNECTION</div>
                        <div class="control-stack compact-stack">
                            <div class="button-row">
                                <button class="action-button command-button" onclick="sendCommand('/input/hdmi1')">HDMI 1</button>
                                <button class="action-button command-button" onclick="sendCommand('/input/hdmi2')">HDMI 2</button>
                            </div>
                            <div class="button-row">
                                <select id="port-select" class="action-button port-select" style="flex: 1;"></select>
                                <button class="action-button command-button" onclick="reconnectPort()">RECONNECT</button>
                            </div>
                        </div>
                    </article>

                    <article class="control-card accent-slate" style="grid-column: 1 / -1;">
                        <div class="card-label">EVENT LOG</div>
                        <ul id="event-log" class="event-log"></ul>
                    </article>
                </section>
                <div class="watermark">
                    <img src="/static/WSLCOE_logo.png" alt="Watermark">
                </div>
            </main>

            <script>
                const commandNames = {{
                    '/start': 'START command',
                    '/progress/reset': 'RESET PROGRESS command',
                    '/power/on': 'POWER ON command',
                    '/power/off': 'POWER OFF command',
                    '/av/mute/on': 'DISPLAY OFF command',
                    '/av/mute/off': 'DISPLAY ON command',
                    '/input/hdmi1': 'HDMI 1 command',
                    '/input/hdmi2': 'HDMI 2 command'
                }};

                function getThemeSuccessColor() {{
                    return getComputedStyle(document.body).getPropertyValue('--start-green').trim() || '#1f8f4a';
                }}

                function renderProgress(progressStep, progressTotal, description, pipelineStatus = 'idle') {{
                    const segments = document.querySelectorAll('.progress-segment');
                    const successColor = getThemeSuccessColor();
                    const themeStyles = getComputedStyle(document.body);
                    const idleColor = themeStyles.getPropertyValue('--text-main').trim() || '#f2f2f3';
                    const errorColor = themeStyles.getPropertyValue('--danger-text').trim() || '#c56f6f';
                    const isFailed = pipelineStatus === 'failed';
                    const hasPipelineTotal = progressTotal > 0;
                    const visibleSegments = hasPipelineTotal
                        ? Math.min(progressTotal, segments.length)
                        : segments.length;
                    segments.forEach((seg) => {{
                        const idx = Number(seg.dataset.index || 0);
                        seg.style.display = idx <= visibleSegments ? 'block' : 'none';
                        if (idx > visibleSegments) {{
                            return;
                        }}
                        if (!isFailed && idx <= progressStep && progressStep > 0) {{
                            seg.style.background = successColor;
                            seg.style.border = '1px solid ' + successColor;
                        }} else {{
                            seg.style.background = 'rgba(255,255,255,0.06)';
                            seg.style.border = '1px solid rgba(255,255,255,0.04)';
                        }}
                    }});
                    const progressText = document.getElementById('progress-text');
                    const progressDesc = document.getElementById('progress-desc');
                    const progressCheck = document.getElementById('progress-check');

                    // Save current progress so theme toggles can re-render correctly
                    window._currentProgress = {{ step: progressStep, total: progressTotal, desc: description, status: pipelineStatus }};

                    if (isFailed) {{
                        progressText.textContent = 'FAILED';
                        progressText.style.color = errorColor;
                        progressText.style.fontWeight = '700';
                        progressText.style.fontSize = '1rem';

                        progressDesc.textContent = description || 'Pipeline failed.';
                        progressDesc.style.color = errorColor;
                        progressDesc.style.fontWeight = '700';
                        progressDesc.style.fontSize = '1rem';

                        progressCheck.textContent = '';
                        progressCheck.style.display = 'none';
                        progressCheck.style.fontSize = '1rem';
                        progressCheck.style.fontWeight = '700';
                        progressCheck.style.lineHeight = '1';
                    }} else if (progressStep <= 0) {{
                        // Idle state: show IDLE and use primary text color (white in dark mode)
                        progressText.textContent = 'IDLE';
                        progressText.style.color = idleColor;
                        progressText.style.fontWeight = '700';
                        progressText.style.fontSize = '1rem';

                        progressDesc.textContent = '';
                        progressDesc.style.color = idleColor;
                        progressDesc.style.fontWeight = '500';
                        progressDesc.style.fontSize = '1rem';

                        progressCheck.textContent = '';
                        progressCheck.style.display = 'none';
                        progressCheck.style.fontSize = '1rem';
                        progressCheck.style.fontWeight = '700';
                        progressCheck.style.lineHeight = '1';
                    }} else {{
                        // Active steps: show STEP and description (description prefixed with em-dash)
                        progressText.textContent = `STEP (${{progressStep}}/${{progressTotal}})`;
                        progressText.style.color = successColor;
                        progressText.style.fontWeight = '700';
                        progressText.style.fontSize = '1rem';

                        if (description) {{
                            // show description without leading dash
                            progressDesc.textContent = `${{description}}`;
                        }} else {{
                            progressDesc.textContent = '';
                        }}
                        progressDesc.style.color = successColor;
                        // Match font-weight with progress label for consistency
                        progressDesc.style.fontWeight = '700';
                        progressDesc.style.fontSize = '1rem';

                        progressCheck.textContent = '✓';
                        progressCheck.style.display = 'inline-flex';
                        progressCheck.style.color = successColor;
                        progressCheck.style.fontSize = '1rem';
                        progressCheck.style.fontWeight = '700';
                        progressCheck.style.lineHeight = '1';
                    }}

                    // Keep all status-value text white (use primary text color) regardless of progress
                    const statusVals = document.querySelectorAll('.status-value');
                    statusVals.forEach((el) => {{
                        el.style.color = idleColor;
                        // Keep weight consistent (600 for values)
                        el.style.fontWeight = '600';
                    }});
                }}

                function setButtonsEnabled(enabled) {{
                    document.querySelectorAll('.command-button').forEach((button) => {{
                        button.disabled = !enabled;
                    }});
                }}

                function handleUnauthorized(response) {{
                    if (response.status === 401) {{
                        window.location.href = '/login';
                        return true;
                    }}
                    return false;
                }}

                function setTheme(theme) {{
                    document.body.setAttribute('data-theme', theme);
                    localStorage.setItem('projector-theme', theme);
                    const toggle = document.getElementById('theme-toggle');
                    const sunIcon = toggle.querySelector('.toggle-sun');
                    const moonIcon = toggle.querySelector('.toggle-moon');
                    toggle.setAttribute('aria-checked', theme === 'dark' ? 'true' : 'false');
                    if (theme === 'dark') {{
                        toggle.classList.remove('light-mode');
                        sunIcon.classList.remove('active');
                        moonIcon.classList.add('active');
                    }} else {{
                        toggle.classList.add('light-mode');
                        sunIcon.classList.add('active');
                        moonIcon.classList.remove('active');
                    }}
                    refreshFeedbackColor();
                    // Re-render progress/status so CSS variable changes take effect immediately
                    const current = window._currentProgress || {{ step: 0, total: 7, desc: '', status: 'idle' }};
                    renderProgress(current.step, current.total, current.desc, current.status);
                }}

                function refreshFeedbackColor() {{
                    const feedbackText = document.getElementById('feedback-text');
                    const feedbackCheck = document.getElementById('feedback-check');
                    const feedbackMessage = document.getElementById('feedback-message');
                    if (!feedbackText) {{
                        return;
                    }}

                    const result = feedbackText.dataset.result;
                    if (result !== 'success' && result !== 'error') {{
                        return;
                    }}

                    const themeStyles = getComputedStyle(document.body);
                    const successColor = getThemeSuccessColor();
                    const errorColor = themeStyles.getPropertyValue('--danger-text').trim() || '#c56f6f';
                    const chosen = result === 'success' ? successColor : errorColor;
                    feedbackText.style.color = chosen;
                    feedbackText.style.fontWeight = '700';
                    if (feedbackCheck) {{
                        feedbackCheck.style.color = chosen;
                        feedbackCheck.style.fontWeight = '700';
                        feedbackCheck.style.fontSize = '1rem';
                        feedbackCheck.style.lineHeight = '1';
                    }}
                    if (feedbackMessage) {{
                        feedbackMessage.style.color = chosen;
                        feedbackMessage.style.fontWeight = '700';
                    }}

                    // Also update progress caption and status text to match the feedback color or idle
                    const current = window._currentProgress || {{ step: 0, total: 7, desc: '', status: 'idle' }};
                    if (current.step > 0) {{
                        // re-render progress to apply green shades
                        renderProgress(current.step, current.total, current.desc, current.status);
                    }} else {{
                        // idle - ensure colors are primary text
                        renderProgress(0, current.total, current.desc, current.status);
                    }}
                }}

                function initTheme() {{
                    const stored = localStorage.getItem('projector-theme');
                    const theme = stored || 'dark';
                    setTheme(theme);
                }}

                function toggleTheme() {{
                    const current = document.body.getAttribute('data-theme') || 'dark';
                    setTheme(current === 'dark' ? 'light' : 'dark');
                }}

                document.getElementById('theme-toggle').addEventListener('click', toggleTheme);

                async function sendCommand(path, confirmMessage = null) {{
                    if (confirmMessage && !window.confirm(confirmMessage)) {{
                        return;
                    }}

                    const commandName = commandNames[path] || 'Command';
                    showFeedback(`Sending ${{commandName}}...`, true);
                    setButtonsEnabled(false);
                    try {{
                        const response = await fetch(path, {{ method: 'POST' }});
                        if (handleUnauthorized(response)) {{
                            return;
                        }}
                        const data = await response.json();
                        if (response.ok && data.success) {{
                            showFeedback(`${{commandName}} sent`, true);
                            if (path === '/start') {{
                                updateStatus();
                            }}
                        }} else {{
                            const reason = data.detail ? ` - ${{data.detail}}` : '';
                            showFeedback(`${{commandName}} not sent${{reason}}`, false);
                        }}
                    }} catch (error) {{
                        showFeedback(`Command failed: ${{error.message}}`, false);
                    }} finally {{
                        setButtonsEnabled(true);
                    }}
                    updateStatus();
                    fetchEvents();
                }}

                async function reconnectPort() {{
                    const select = document.getElementById('port-select');
                    const port = select.value;

                    if (!port) {{
                        showFeedback('No COM port selected', false);
                        return;
                    }}

                    showFeedback(`Connecting to ${{port}}...`, true);
                    setButtonsEnabled(false);
                    try {{
                        const response = await fetch('/connect', {{
                            method: 'POST',
                            headers: {{ 'Content-Type': 'application/json' }},
                            body: JSON.stringify({{ port }})
                        }});
                        if (handleUnauthorized(response)) {{
                            return;
                        }}
                        const data = await response.json();
                        if (response.ok && data.success) {{
                            showFeedback(`Connected to ${{port}}`, true);
                        }} else {{
                            showFeedback(`Connect failed${{data.detail ? ` - ${{data.detail}}` : ''}}`, false);
                        }}
                    }} catch (error) {{
                        showFeedback(`Connect failed: ${{error.message}}`, false);
                    }} finally {{
                        setButtonsEnabled(true);
                    }}

                    updateStatus();
                    fetchPorts();
                    fetchEvents();
                }}

                function showFeedback(message, success) {{
                    const feedbackText = document.getElementById('feedback-text');
                    const feedbackCheck = document.getElementById('feedback-check');
                    const feedbackMessage = document.getElementById('feedback-message');
                    const themeStyles = getComputedStyle(document.body);
                    const successColor = getThemeSuccessColor();
                    const errorColor = themeStyles.getPropertyValue('--danger-text').trim() || '#c56f6f';
                    const now = new Date().toLocaleTimeString();
                    if (feedbackMessage) {{
                        feedbackMessage.textContent = `${{message}} at ${{now}}`;
                    }} else {{
                        feedbackText.textContent = `${{success ? '✓ ' : ''}}${{message}} at ${{now}}`;
                    }}
                    if (feedbackCheck) {{
                        feedbackCheck.textContent = success ? '✓' : '';
                        feedbackCheck.style.display = success ? 'inline-flex' : 'none';
                    }}
                    feedbackText.dataset.result = success ? 'success' : 'error';
                    feedbackText.style.color = success ? successColor : errorColor;
                    feedbackText.style.fontWeight = '700';
                    if (feedbackCheck) {{
                        feedbackCheck.style.color = success ? successColor : errorColor;
                        feedbackCheck.style.fontWeight = '700';
                        feedbackCheck.style.fontSize = '1rem';
                        feedbackCheck.style.lineHeight = '1';
                    }}
                    if (feedbackMessage) {{
                        feedbackMessage.style.color = success ? successColor : errorColor;
                        feedbackMessage.style.fontWeight = '700';
                    }}
                }}

                async function fetchPorts() {{
                    try {{
                        const response = await fetch('/ports');
                        if (handleUnauthorized(response)) {{
                            return;
                        }}
                        const data = await response.json();
                        const select = document.getElementById('port-select');
                        const current = data.current_port || '';

                        select.innerHTML = '';
                        (data.ports || []).forEach((port) => {{
                            const option = document.createElement('option');
                            option.value = port;
                            option.textContent = port;
                            if (port === current) {{
                                option.selected = true;
                            }}
                            select.appendChild(option);
                        }});

                        if (!select.value && current) {{
                            const option = document.createElement('option');
                            option.value = current;
                            option.textContent = current;
                            option.selected = true;
                            select.appendChild(option);
                        }}
                    }} catch (error) {{
                        console.error('Port fetch failed:', error);
                    }}
                }}

                async function fetchEvents() {{
                    try {{
                        const response = await fetch('/events');
                        if (handleUnauthorized(response)) {{
                            return;
                        }}
                        const data = await response.json();
                        const log = document.getElementById('event-log');
                        log.innerHTML = '';

                        (data.events || []).slice(-10).reverse().forEach((entry) => {{
                            const item = document.createElement('li');
                            item.className = entry.success ? 'event-ok' : 'event-fail';
                            item.textContent = `[${{entry.time}}] ${{entry.message}}`;
                            log.appendChild(item);
                        }});
                    }} catch (error) {{
                        console.error('Event fetch failed:', error);
                    }}
                }}

                async function updateStatus() {{
                    try {{
                        const response = await fetch('/status');
                        if (handleUnauthorized(response)) {{
                            return;
                        }}
                        const data = await response.json();
                        document.getElementById('status-power').textContent = data.power || 'unknown';
                        document.getElementById('status-input').textContent = data.input || 'unknown';
                        document.getElementById('status-connection').textContent = data.connection_detail || 'unknown';
                        document.getElementById('status-time').textContent = data.last_update || '--';
                        const badge = document.querySelector('.status-badge');
                        badge.textContent = data.connected ? 'Connected' : 'Disconnected';
                        badge.classList.toggle('connected', !!data.connected);
                        badge.classList.toggle('disconnected', !data.connected);
                        // Render pipeline progress if present
                        const pStep = Number(data.progress_step || 0);
                        const pTotal = Number(data.progress_total || 0);
                        const pDesc = data.progress_description || '';
                        const pStatus = data.pipeline_status || 'idle';
                        renderProgress(pStep, pTotal, pDesc, pStatus);
                        // Enable Save Result button when pipeline is completed
                        const saveBtn = document.getElementById('save-result-btn');
                        if (data.pipeline_status === 'completed') {{
                            saveBtn.disabled = false;
                        }} else {{
                            saveBtn.disabled = true;
                        }}
                    }} catch (error) {{
                        console.error('Status fetch failed:', error);
                        showFeedback(`Status refresh failed: ${{error.message}}`, false);
                    }}
                }}

                // Poll status frequently so pipeline updates feel live.
                const statusInterval = setInterval(updateStatus, 1000);

                function toggleProfileMenu() {{
                    const dropdown = document.getElementById('profile-dropdown');
                    dropdown.classList.toggle('visible');
                }}

                // Close profile menu when clicking outside
                document.addEventListener('click', (event) => {{
                    const profileMenu = document.getElementById('profile-menu');
                    if (profileMenu && !profileMenu.contains(event.target)) {{
                        const dropdown = document.getElementById('profile-dropdown');
                        dropdown.classList.remove('visible');
                    }}
                }});

                async function logout() {{
                    const response = await fetch('/logout', {{ method: 'POST' }});
                    if (response.redirected) {{
                        window.location.href = '/login';
                    }}
                }}

                async function shutdownApp() {{
                    if (!window.confirm('Are you sure you want to shutdown the application?')) {{
                        return;
                    }}

                    try {{
                        const response = await fetch('/shutdown', {{ method: 'POST' }});
                        if (handleUnauthorized(response)) {{
                            return;
                        }}
                        const data = await response.json();
                        if (!response.ok || !data.success) {{
                            showFeedback(`Shutdown failed${{data.detail ? ` - ${{data.detail}}` : ''}}`, false);
                            return;
                        }}
                        showFeedback('Shutting down...', true);
                        setButtonsEnabled(false);
                        setTimeout(() => {{
                            clearInterval(statusInterval);
                            document.title = 'Shutdown Complete - Automatic Decal Projection';
                            document.body.innerHTML = `
                                <main style="min-height:100vh;display:grid;place-items:center;padding:24px;background:#0b0c0d;color:#f2f2f3;font-family:'Segoe UI',Arial,sans-serif;">
                                    <section style="max-width:520px;width:100%;border:1px solid rgba(154,154,154,.25);border-radius:10px;padding:24px;background:rgba(31,33,36,.96);box-shadow:0 14px 44px rgba(0,0,0,.52);">
                                        <div style="display:grid;place-items:center;margin-bottom:10px;">
                                            <img src="/static/fontaine-logo-white.png" alt="Automatic Decal Projection logo" style="width:156px;height:auto;">
                                        </div>
                                        <h1 style="margin:0 0 10px;font-size:1.4rem;">Application Stopped</h1>
                                        <p style="margin:0 0 8px;color:#d0d2d6;">Projector Control has shut down successfully.</p>
                                        <p style="margin:0;color:#a9abb0;">You can close this tab and restart the server when needed.</p>
                                    </section>
                                </main>
                            `;
                        }}, 700);
                    }} catch (error) {{
                        showFeedback(`Shutdown failed: ${{error.message}}`, false);
                    }}
                }}

                async function restartApp() {{
                    if (!window.confirm('Force restart the application? This will stop the pipeline and restart the GUI.')) {{
                        return;
                    }}

                    try {{
                        const response = await fetch('/restart', {{ method: 'POST' }});
                        if (handleUnauthorized(response)) {{
                            return;
                        }}
                        const data = await response.json();
                        if (!response.ok || !data.success) {{
                            showFeedback(`Restart failed${{data.detail ? ` - ${{data.detail}}` : ''}}`, false);
                            return;
                        }}
                        showFeedback('Restarting...', true);
                        setButtonsEnabled(false);
                        setTimeout(() => {{
                            clearInterval(statusInterval);
                            document.title = 'Restarting - Automatic Decal Projection';
                            document.body.innerHTML = `
                                <main style="min-height:100vh;display:grid;place-items:center;padding:24px;background:#0b0c0d;color:#f2f2f3;font-family:'Segoe UI',Arial,sans-serif;">
                                    <section style="max-width:520px;width:100%;border:1px solid rgba(154,154,154,.25);border-radius:10px;padding:24px;background:rgba(31,33,36,.96);box-shadow:0 14px 44px rgba(0,0,0,.52);">
                                        <div style="display:grid;place-items:center;margin-bottom:10px;">
                                            <img src="/static/fontaine-logo-white.png" alt="Automatic Decal Projection logo" style="width:156px;height:auto;">
                                        </div>
                                        <h1 style="margin:0 0 10px;font-size:1.4rem;">Restarting Application</h1>
                                        <p style="margin:0 0 8px;color:#d0d2d6;">Projector Control is shutting down and starting again.</p>
                                        <p id="restart-status" style="margin:0;color:#a9abb0;">Waiting for the GUI to come back online...</p>
                                    </section>
                                </main>
                            `;
                            waitForRestart();
                        }}, 700);
                    }} catch (error) {{
                        showFeedback(`Restart failed: ${{error.message}}`, false);
                    }}
                }}

                async function waitForRestart() {{
                    const status = document.getElementById('restart-status');
                    const started = Date.now();
                    await new Promise(resolve => setTimeout(resolve, 3500));

                    while (Date.now() - started < 45000) {{
                        try {{
                            const response = await fetch('/', {{ method: 'GET', cache: 'no-store' }});
                            if (response.ok) {{
                                window.location.href = '/';
                                return;
                            }}
                        }} catch (error) {{
                            // The server is expected to be offline briefly during restart.
                        }}
                        await new Promise(resolve => setTimeout(resolve, 1500));
                    }}

                    if (status) {{
                        status.textContent = 'Restart is taking longer than expected. Refresh this page in a moment.';
                    }}
                }}

                async function saveResult() {{
                    const saveBtn = document.getElementById('save-result-btn');
                    if (saveBtn.disabled) {{
                        return;
                    }}

                    showFeedback('Saving result...', true);
                    saveBtn.disabled = true;
                    try {{
                        const response = await fetch('/save-result', {{ method: 'POST' }});
                        if (handleUnauthorized(response)) {{
                            return;
                        }}
                        const data = await response.json();
                        if (response.ok && data.success) {{
                            showFeedback(`Result saved successfully`, true);
                        }} else {{
                            showFeedback(`Save failed${{data.detail ? ` - ${{data.detail}}` : ''}}`, false);
                        }}
                    }} catch (error) {{
                        showFeedback(`Save failed: ${{error.message}}`, false);
                    }} finally {{
                        // Keep button disabled until reset
                        updateStatus();
                    }}
                }}

                // Initial status load
                initTheme();
                fetchPorts();
                fetchEvents();
                updateStatus();
            </script>
        </body>
    </html>
    """

@app.post("/power/on")
def power_on(request: Request):
    if not _is_authenticated(request):
        return _api_unauthorized_response()

    if not command_lock.acquire(blocking=False):
        return {"status": "busy", "detail": "command already in progress", "success": False}

    try:
        if projector is None:
            _record_event("POWER ON command not sent (disconnected)", False)
            return {"status": "not sent", "detail": "projector disconnected", "success": False}

        projector.power_on()
        state["power"] = "On"
        state["last_update"] = _timestamp()
        _record_event("POWER ON command sent", True)
        return {"status": "sent", "success": True}
    finally:
        command_lock.release()

@app.post("/power/off")
def power_off(request: Request):
    if not _is_authenticated(request):
        return _api_unauthorized_response()

    if not command_lock.acquire(blocking=False):
        return {"status": "busy", "detail": "command already in progress", "success": False}

    try:
        if projector is None:
            _record_event("POWER OFF command not sent (disconnected)", False)
            return {"status": "not sent", "detail": "projector disconnected", "success": False}

        projector.power_off()
        state["power"] = "Off"
        state["last_update"] = _timestamp()
        _record_event("POWER OFF command sent", True)
        return {"status": "sent", "success": True}
    finally:
        command_lock.release()


@app.post("/av/mute/on")
def av_mute_on(request: Request):
    if not _is_authenticated(request):
        return _api_unauthorized_response()

    if not command_lock.acquire(blocking=False):
        return {"status": "busy", "detail": "command already in progress", "success": False}

    try:
        if projector is None:
            _record_event("DISPLAY OFF command not sent (disconnected)", False)
            return {"status": "not sent", "detail": "projector disconnected", "success": False}

        projector.av_mute_on()
        state["last_update"] = _timestamp()
        _record_event("DISPLAY OFF command sent", True)
        return {"status": "sent", "success": True}
    finally:
        command_lock.release()


@app.post("/av/mute/off")
def av_mute_off(request: Request):
    if not _is_authenticated(request):
        return _api_unauthorized_response()

    if not command_lock.acquire(blocking=False):
        return {"status": "busy", "detail": "command already in progress", "success": False}

    try:
        if projector is None:
            _record_event("DISPLAY ON command not sent (disconnected)", False)
            return {"status": "not sent", "detail": "projector disconnected", "success": False}

        projector.av_mute_off()
        state["last_update"] = _timestamp()
        _record_event("DISPLAY ON command sent", True)
        return {"status": "sent", "success": True}
    finally:
        command_lock.release()


@app.post("/input/hdmi1")
def input_hdmi1(request: Request):
    if not _is_authenticated(request):
        return _api_unauthorized_response()

    if not command_lock.acquire(blocking=False):
        return {"status": "busy", "detail": "command already in progress", "success": False}

    try:
        if projector is None:
            _record_event("HDMI 1 command queued (disconnected)", False)
            return {"status": "not sent", "detail": "projector disconnected", "success": False}

        projector.hdmi_1()
        state["input"] = "HDMI 1"
        state["last_update"] = _timestamp()
        _record_event("HDMI 1 command sent", True)
        return {"status": "sent", "success": True}
    finally:
        command_lock.release()


@app.post("/input/hdmi2")
def input_hdmi2(request: Request):
    if not _is_authenticated(request):
        return _api_unauthorized_response()

    if not command_lock.acquire(blocking=False):
        return {"status": "busy", "detail": "command already in progress", "success": False}

    try:
        if projector is None:
            _record_event("HDMI 2 command queued (disconnected)", False)
            return {"status": "not sent", "detail": "projector disconnected", "success": False}

        projector.hdmi_2()
        state["input"] = "HDMI 2"
        state["last_update"] = _timestamp()
        _record_event("HDMI 2 command sent", True)
        return {"status": "sent", "success": True}
    finally:
        command_lock.release()

def _pipeline_progress_callback(step: int, total: int, description: str) -> None:
    _set_progress(
        step=step,
        total=total,
        description=description,
        pipeline_running=True,
        pipeline_status="running",
    )


def _run_pipeline() -> None:
    try:
        from run import main as run_pipeline

        run_pipeline(progress_callback=_pipeline_progress_callback)
        _set_progress(
            pipeline_running=False,
            pipeline_status="completed",
        )
        _record_event("Pipeline completed", True)
    except Exception as exc:
        _set_progress(
            description=f"Failed: {exc}",
            pipeline_running=False,
            pipeline_status="failed",
        )
        _record_event(f"Pipeline failed: {exc}", False)
    finally:
        if pipeline_lock.locked():
            pipeline_lock.release()


def _idle_projector_window() -> None:
    from homography.door_feature_helper import clear_model_cache
    import homography.final_projection_pipeline as base

    if base.PROJECTOR_WINDOW.is_open:
        idle_canvas = np.full(
            (base.PROJECTOR_HEIGHT, base.PROJECTOR_WIDTH, 3),
            base.SCENE_CANVAS_COLOR,
            dtype=np.uint8,
        )
        base.PROJECTOR_WINDOW.show(idle_canvas, label="idle projector output")
        base.PROJECTOR_WINDOW.wait_for_settle(1)

    # Clear the cached YOLO model to prevent hang on next pipeline run.
    clear_model_cache()


@app.post("/start")
def start_program(request: Request):
    if not _is_authenticated(request):
        return _api_unauthorized_response()

    if not pipeline_lock.acquire(blocking=False):
        return {"status": "busy", "detail": "pipeline already running", "success": False}

    try:
        _set_progress(
            step=0,
            total=0,
            description="Starting pipeline...",
            pipeline_running=True,
            pipeline_status="running",
        )
        _record_event("START command sent", True)
        if projector is not None:
            projector.av_mute_off()
        pipeline_executor.submit(_run_pipeline)
        return {"status": "program started", "success": True}
    except Exception:
        _set_progress(
            description="Failed: Could not start pipeline.",
            pipeline_running=False,
            pipeline_status="failed",
        )
        if pipeline_lock.locked():
            pipeline_lock.release()
        raise


@app.get("/ports")
def get_ports(request: Request):
    if not _is_authenticated(request):
        return _api_unauthorized_response()

    ports = [port.device for port in list_ports.comports()]
    if current_port and current_port not in ports:
        ports.append(current_port)
    return {"ports": sorted(ports), "current_port": current_port}


@app.post("/connect")
def connect_projector(request: Request, connect_request: ConnectRequest):
    if not _is_authenticated(request):
        return _api_unauthorized_response()

    ok = _connect_projector(connect_request.port)
    return {
        "success": ok,
        "status": "connected" if ok else "failed",
        "detail": None if ok else (serial_error or "connection failed"),
    }


@app.get("/events")
def get_events(request: Request):
    if not _is_authenticated(request):
        return _api_unauthorized_response()

    return {"events": event_log[-40:]}

@app.get("/status")
def get_status(request: Request):
    if not _is_authenticated(request):
        return _api_unauthorized_response()

    state["port"] = current_port
    if not state["connected"]:
        state["power"] = "Off"
    state["connection_detail"] = _connection_detail()
    if state["last_update"] == "--":
        state["last_update"] = _timestamp()
    return state


@app.post("/progress/set")
def set_progress(request: Request, payload: ProgressRequest):
    if not _is_authenticated(request):
        return _api_unauthorized_response()

    _set_progress(
        step=payload.step,
        total=payload.total,
        description=payload.description or "",
    )
    # Do NOT record progress updates in the event log (event_log is for commands only)
    return {"success": True, "step": state["progress_step"], "total": state["progress_total"]}


@app.post("/progress/reset")
def reset_progress(request: Request):
    if not _is_authenticated(request):
        return _api_unauthorized_response()

    if pipeline_lock.locked() or state.get("pipeline_running"):
        return JSONResponse(
            {"success": False, "status": "busy", "detail": "pipeline is still running"},
            status_code=409,
        )

    if projector is not None:
        projector.av_mute_on()
    _set_progress(
        step=0,
        total=0,
        description="IDLE",
        pipeline_running=False,
        pipeline_status="idle",
    )
    _record_event("Pipeline progress reset", True)

    cleanup_future = pipeline_executor.submit(_idle_projector_window)
    try:
        cleanup_future.result(timeout=5)
    except FutureTimeoutError:
        _record_event("Projector window reset timed out", False)
        return JSONResponse(
            {"success": False, "status": "failed", "detail": "projector window reset timed out"},
            status_code=500,
        )
    except Exception as exc:
        _record_event(f"Projector window reset failed: {exc}", False)
        return JSONResponse(
            {"success": False, "status": "failed", "detail": str(exc)},
            status_code=500,
        )
    return {"success": True, "status": "reset"}


@app.post("/save-result")
def save_result(request: Request):
    if not _is_authenticated(request):
        return _api_unauthorized_response()

    if pipeline_lock.locked() or state.get("pipeline_running"):
        return JSONResponse(
            {"success": False, "detail": "pipeline is still running"},
            status_code=409,
        )

    try:
        from homography.final_projection_pipeline import open_camera

        # Capture image from camera (device 0)
        with open_camera(width=1920, height=1080) as cap:
            # Set camera properties for better quality
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Read several frames to let camera warm up and autofocus
            # Discard first few frames as they are often corrupted or incomplete
            frame = None
            for _ in range(10):
                ret, frame = cap.read()
                if not ret:
                    return JSONResponse(
                        {"success": False, "detail": "Could not capture frames from camera"},
                        status_code=400
                    )
        
        if frame is None or frame.size == 0:
            return JSONResponse(
                {"success": False, "detail": "Captured frame is invalid"},
                status_code=400
            )
        
        # Rotate 90 degrees clockwise (cv2.rotate uses clockwise for ROTATE_90_CLOCKWISE)
        rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        # Create complete_jobs directory if it doesn't exist
        complete_jobs_dir = Path.home() / "Documents" / "Final_Pipeline" / "complete_jobs"
        complete_jobs_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"result_{timestamp}.jpg"
        filepath = complete_jobs_dir / filename
        
        # Save the rotated image
        cv2.imwrite(str(filepath), rotated)
        
        _record_event(f"Saved result image: {filename}", True)
        return {
            "success": True,
            "filename": filename,
            "path": str(filepath)
        }
    except Exception as exc:
        _record_event(f"Failed to save result: {exc}", False)
        return JSONResponse(
            {"success": False, "detail": str(exc)},
            status_code=400
        )

@app.on_event("shutdown")
def shutdown_event():
    _shutdown_runtime_resources()
