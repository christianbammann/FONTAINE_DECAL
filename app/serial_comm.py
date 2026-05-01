# This code will interface with the projector and send the actual command packets using pyserial.

import serial

class ProjectorSerial:
    def __init__(self, port: str):
        self.ser = serial.Serial(
            port=port,
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        )
        print(f"Serial connection established on port {port}")

    def send_bytes(self, data: bytes):
        print("Bytes to send:", data.hex())
        self.ser.write(data)

    def close(self):
        self.ser.close()
