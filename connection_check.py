import os
import sys
import subprocess
import time

def connection_checking(connection=True):
    cmd_line = 'who -s' # list ssh sessions connected with 192 value to connect from
    spell = subprocess.Popen([cmd_line], stdout=subprocess.PIPE, shell=True)
    (out, _) = spell.communicate()
    # print(out)
    numbers_of_remote_ip = out.decode("utf-8").split()
    # print(numbers_of_remote_ip)
    matching = [ips for ips in numbers_of_remote_ip if "(192." in ips]
    # print(matching)
    
    if int(len(matching)) >= 1:
        if connection == False:
            print("Reconnected to the System")
            connection = True
    else:
        connection = False
    
    return connection
