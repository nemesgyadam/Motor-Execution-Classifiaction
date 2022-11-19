import subprocess


def list_ssids():
    """
    List all available SSIDs
    """
    out = subprocess.check_output("netsh wlan show interfaces").decode("utf-8")
    print("The computer is currently connected to the following SSIDs:")
    ssids = []
    for line in out.split("\n"):
        if "SSID" in line and "BSSID" not in line:
            print(line)
            ssids.append(line.split(":")[1].strip())
    return ssids
