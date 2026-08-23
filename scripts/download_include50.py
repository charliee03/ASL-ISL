import socket
import subprocess
import time
import sys

# Monkeypatch socket.getaddrinfo to resolve via 8.8.8.8 when local WSL DNS fails
_old_getaddrinfo = socket.getaddrinfo
_dns_cache = {}

def _dns_lookup_8888(hostname):
    if hostname in _dns_cache:
        return _dns_cache[hostname]
    try:
        out = subprocess.check_output(['host', hostname, '8.8.8.8'], text=True, timeout=5)
        for line in out.splitlines():
            if 'has address' in line:
                ip = line.split()[-1]
                _dns_cache[hostname] = ip
                return ip
    except Exception:
        pass
    return None

def custom_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    try:
        return _old_getaddrinfo(host, port, family, type, proto, flags)
    except socket.gaierror:
        ip = _dns_lookup_8888(host)
        if ip:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (ip, port))]
        raise

socket.getaddrinfo = custom_getaddrinfo

import kagglehub

def main():
    max_retries = 50
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempt {attempt}/{max_retries}: Downloading yuvrajjoshi1110/include-50 via kagglehub...")
            path = kagglehub.dataset_download("yuvrajjoshi1110/include-50")
            print("Successfully downloaded! Path to dataset files:", path)
            return
        except Exception as e:
            print(f"Error on attempt {attempt}: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    main()
