import socket
import pickle
from importlib import reload

s = socket.socket()
host = socket.gethostbyaddr('ip-172-31-2-21.us-east-2.compute.internal')[0]
port = 28
s.bind((host, port))

def recvall(sock):
    BUFF_SIZE = 4096
    data = b''
    while True:
        part = sock.recv(BUFF_SIZE)
        data += part
        if part.endswith(b'#EOF'):
            break
    return data

s.listen(5)
while True:
    c, addr = s.accept()
    print('Got connection from', addr)
    data = recvall(c)
    with open('script.py', 'wb') as f:
        f.write(data)
    print('Received script.py')

    try:
        try: reload(script)
        except Exception: import script
        results = script.main()

        c.sendall(pickle.dumps(results))
        c.send('#EOF'.encode())
    except Exception as e:
        print(e)
        c.sendall(pickle.dumps(e))
        c.send('#EOF'.encode())

    c.close()