import socket
import winsound
import pickle

s = socket.socket()
host = socket.gethostbyaddr('3.139.238.226')[0]
port = 28

s.connect((host, port))
with open("./../nevergiveup.py", "rb") as f:
    data = f.read()
    s.sendall(data)
    s.send("#EOF".encode())

def recvall(sock):
    BUFF_SIZE = 4096
    data = b''
    while True:
        part = sock.recv(BUFF_SIZE)
        data += part
        if part.endswith(b'#EOF'):
            break
    return data

data = pickle.loads(recvall(s))
print(data)
winsound.Beep(500, 500)

s.close()
print("breakpoint")