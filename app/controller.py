# This code only contains the serial commands in byte form to act as a translator.
# Also stored on team Google Drive in the Elec Schematics folder.

from app.serial_comm import ProjectorSerial

# General commands 
RESET_ALL = bytes.fromhex("06 14 00 04 00 34 11 02 00 5F") # resets ALL settings
POWER_ON_CMD = bytes.fromhex("06 14 00 04 00 34 11 00 00 5D")
POWER_OFF_CMD = bytes.fromhex("06 14 00 04 00 34 11 01 00 5E")
AV_MUTE_ON = bytes.fromhex("06 14 00 04 00 34 12 09 01 68")
AV_MUTE_OFF = bytes.fromhex("06 14 00 04 00 34 12 09 00 67")    

# Source inputs
HDMI_1 = bytes.fromhex("06 14 00 04 00 34 13 01 03 63")
HDMI_2 = bytes.fromhex("06 14 00 04 00 34 13 01 07 67")

# Remote inputs
REM_MENU = bytes.fromhex("06 14 00 04 00 34 02 04 0F 61")
REM_ENTER = bytes.fromhex ("06 14 00 04 00 34 02 04 13 65")
REM_RIGHT = bytes.fromhex("06 14 00 04 00 34 02 04 0E 60")
REM_UP = bytes.fromhex("06 14 00 04 00 34 02 04 0B 5D")

class ProjectorController:
    def __init__(self, serial_iface: ProjectorSerial):
        self.serial = serial_iface

    def power_on(self):
        print("Power ON command triggered")
        self.serial.send_bytes(POWER_ON_CMD)

    def power_off(self):
        self.serial.send_bytes(POWER_OFF_CMD)
    
    def av_mute_on(self):
        self.serial.send_bytes(AV_MUTE_ON)

    def av_mute_off(self):
        self.serial.send_bytes(AV_MUTE_OFF)

    def hdmi_1(self):
        self.serial.send_bytes(HDMI_1)

    def hdmi_2(self):
        self.serial.send_bytes(HDMI_2)