import paramiko
import os
from cryptography.fernet import Fernet
import json

class Conn2Remote(object):
    
    def __init__(self, host, user, key_filename, port=22, passcode=None):
        self.host = host
        self.user = user
        self.key_filename = key_filename
        self.passcode = passcode
        self.port = port
        self.con = False
        if passcode: self.decrypt()
        self.conn()
        return
    
    def decrypt(self):
        passcode = bytes(self.passcode, encoding="utf8")
        cipher_suite = Fernet(passcode)
        self.user = cipher_suite.decrypt(bytes(self.user, encoding="utf8")).decode("utf-8")
        self.host = cipher_suite.decrypt(bytes(self.host, encoding="utf8")).decode("utf-8")
        return
    
    def conn(self):
        if not self.con:
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh.connect(hostname=self.host, port = self.port, username=self.user, key_filename=self.key_filename)
            self.scp = paramiko.SFTPClient.from_transport(self.ssh.get_transport())
            self.con = True
        return
    
    def close(self):
        if self.con:
            self.scp.close()
            self.ssh.close()
        return
    
def encrypt(host, user, filename="config/passcode.json"):
    passcode = Fernet.generate_key()
    cipher_suite = Fernet(passcode)
    host = cipher_suite.encrypt(bytes(host, encoding="utf8"))
    user = cipher_suite.encrypt(bytes(user, encoding="utf8"))
    with open(filename, "w") as f:
        f.write(json.dumps({"user": user.decode("utf-8"), "host": host.decode("utf-8"), "passcode": passcode.decode("utf-8")},
                           sort_keys=True, indent=4))
    return

def get_session(filename="config/passcode.json", key_filename="", isclose=False):
    with open(filename, "r") as f:
        obj = json.loads("".join(f.readlines()))
        conn = Conn2Remote(obj["host"], obj["user"], 
                           key_filename=key_filename, 
                           passcode=obj["passcode"])
    if isclose: conn.close()    
    return conn

def chek_remote_file_exists(fname, conn):
    try:
        print("File Check:",fname)
        conn.scp.stat(fname)
        return True
    except FileNotFoundError:
        return False

def to_remote_FS(conn, local_file, LFS, is_local_remove=False):
    remote_file = LFS + local_file.replace("../", "")
    print(" To file:", remote_file)
    conn.scp.put(local_file, remote_file)
    if is_local_remove: os.remove(local_file)
    return

def from_remote_FS(conn, local_file, LFS):
    remote_file = LFS + local_file.replace("../", "")
    print(" From file:", remote_file)
    conn.scp.get(remote_file, local_file)
    return

def to_remote_FS_dir(conn, ldir, local_file, LFS, is_local_remove=False):
    remote_file = LFS + local_file.replace("../", "")
    rdir = LFS + ldir.replace("../", "")
    conn.ssh.exec_command("mkdir -p " + rdir)
    print(" To file:", remote_file)
    conn.scp.put(local_file, remote_file)
    if is_local_remove: os.remove(local_file)
    return

def get_pubfile():
    with open("config/pub.json", "r") as f:
        obj = json.loads("".join(f.readlines()))
        pubfile = obj["pubfile"]
    return pubfile

def fetch_file(conn, local_file, LFS):
    remote_file = LFS + local_file.replace("../", "")
    is_remote = chek_remote_file_exists(remote_file, conn)
    if is_remote: from_remote_FS(conn, local_file, LFS)
    return is_remote