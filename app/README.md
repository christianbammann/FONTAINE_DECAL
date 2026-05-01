=================================================

Univeristy of North Carolina at Charlotte
William States Lee College of Engineering
Senior Design II - FONTAINE_DECAL
Authors: Ryan Monroe, Christian Bammann

=================================================

History:
    01/21/26 - Created
    02/09/26 - Added power on/off functionality
    04/07/26 - Revamped color scheme, added start, status, power, connection, and event log feaatures
    4/08/26 - Added login and shutdown options
    4/09/26 - Added logos and branding

=================================================

File structure:
    __init__.py - Python package indicator
    gui.py - web based user interface (FastAPI)
    controller.py - abstracted projector commands
    serial_comm.py - RS-232 serial communication
    requirements.txt - dependencies list

=================================================

Startup Process:
    Use the following terminal command in the "user-interface" directory

    Start-Process "http://127.0.0.1:8000"; .\.venv\Scripts\python.exe -m uvicorn app.gui:app --host 127.0.0.1 --port 8000 --reload

    uvicorn app.gui:app --host 10.103.224.52 --port 8000 # To host web server manually

    lsof -ti:8000 | xargs kill -15 # To end process started automatically
    
=================================================