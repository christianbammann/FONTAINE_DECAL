APP_DIR="/home/fontaine-decal/Documents/Final_Pipeline"

echo "Startup script ran at $(date)" >> "$APP_DIR/startup.log"
exec >> "$APP_DIR/terminal.log" 2>&1
echo "GUI terminal output started at $(date)"

cd "$APP_DIR"
uvicorn app.gui:app --host 0.0.0.0 --port 8000
