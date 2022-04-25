import multiprocessing.pool
import functools
import serial


class Lego():
    def __init__(self,port="COM6"):
        
        self.ser = serial.Serial(port,9600)
        self.ser.write(b'\x03')
    
    def command(self, py_command):
        py_command = py_command + '\r\n'
        py_command = py_command.encode('utf-8')
        self.ser.write(py_command)
    
    def read_print(self,stop_string):
        return self.ser.read_until(stop_string.encode('utf-8')).decode('utf-8')
  
    




