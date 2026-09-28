"""Fail visibly if the mathematical test process attempts external networking.

Loaded only when this directory is explicitly placed on PYTHONPATH by the
Lagrange mathematical-suite SLURM script. Local loopback and Unix-domain
sockets remain available; external DNS lookups and connections are blocked.
"""

import ipaddress
import os
import socket
import sys


QSO_NETWORK_GUARD_ACTIVE = True


def _local_host(host):
    if host in (None, "", "localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except (TypeError, ValueError):
        return False


def _reject(operation, target):
    message = f"QSO_NETWORK_BLOCKED {operation}: {target!r}"
    audit_path = os.environ.get("QSO_NETWORK_AUDIT_LOG")
    if audit_path:
        with open(audit_path, "a", encoding="utf-8") as audit:
            audit.write(message + "\n")
    print(message, file=sys.stderr, flush=True)
    raise RuntimeError(message)


_original_getaddrinfo = socket.getaddrinfo


def _guarded_getaddrinfo(host, *args, **kwargs):
    if not _local_host(host):
        _reject("getaddrinfo", host)
    return _original_getaddrinfo(host, *args, **kwargs)


socket.getaddrinfo = _guarded_getaddrinfo

_original_gethostbyname = socket.gethostbyname
_original_gethostbyname_ex = socket.gethostbyname_ex


def _guarded_gethostbyname(host):
    if not _local_host(host):
        _reject("gethostbyname", host)
    return _original_gethostbyname(host)


def _guarded_gethostbyname_ex(host):
    if not _local_host(host):
        _reject("gethostbyname_ex", host)
    return _original_gethostbyname_ex(host)


socket.gethostbyname = _guarded_gethostbyname
socket.gethostbyname_ex = _guarded_gethostbyname_ex


_original_connect = socket.socket.connect
_original_connect_ex = socket.socket.connect_ex


def _guarded_connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        if not isinstance(address, tuple) or not _local_host(address[0]):
            _reject("connect", address)
    return _original_connect(self, address)


def _guarded_connect_ex(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        if not isinstance(address, tuple) or not _local_host(address[0]):
            _reject("connect_ex", address)
    return _original_connect_ex(self, address)


socket.socket.connect = _guarded_connect
socket.socket.connect_ex = _guarded_connect_ex
