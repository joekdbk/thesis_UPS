from pymodbus.client import ModbusTcpClient
from pymodbus.payload import BinaryPayloadDecoder
from pymodbus.constants import Endian
from pymodbus.payload import BinaryPayloadBuilder
import time
import threading
#Steps:
# verify conenction
#USe coil 2 to reset if needed (write 1)
#read 30029 to check the bess state (must return 1)
#Read 30033, must be 65535 for a healthy system

#To start:
"""
1. Start Command (Coil):
o Write 1 to Coil 00000 (BESS_START_REQ_R)
2. Monitor State:
o Read Input Register 30029 (BESS_STATE)
o Expected transition:
o 1 → 2 →4→ 6
o Stopped → Starting → Battery On → Running in Following
3. Set Inverter Output Commands
o Use Holding Registers 40002 through 40005 to set the inverters Real and
Reactive Power (in Following), and Voltage/Frequency Reference
Setpoints(in Forming)
1. Voltage and Frequency setpoints will be critical for the
transition transient
Transitioning to FORMING Mode
Once the inverter is running in FOLLOWING mode:
1. Open the Islanding Breaker
o This should cause the inverter to automatically transition to Forming
2. Signal Breaker Open to Inverter
o Write 0 to Coil 00004 (BESS_GRID_RECONNECT)
 0 = Breaker Open
 1 = Breaker Closed (Grid Connected)

3. Automatic Transition Occurs

o Inverter detects grid disconnection and shifts to Forming mode.
4. Monitor Transition:
o Read 30029 (BESS_STATE)
o Confirm transition:
o 6 → 5
o RUNNING in FOLLOWING → RUNNING in FORMING
"""



# Device parameters
IP_ADDRESS = '100.66.19.3' #INPUT YOUR IP ADDRESS HERE
PORT = 1025 # given by KORE
UNIT_ID = 1 #should be 1 since we're using TCP

# Create Modbus client and connect
client = ModbusTcpClient(IP_ADDRESS, port=PORT)
if not client.connect():
    print("Failed to connect to device.")
    exit(1)

client.connect()

def keep_alive(client, UNIT_ID):
    while True:
        try:
            client.read_coils(address=0, count=1, slave=UNIT_ID)  # Read a dummy coil
        except Exception as e:
            print("Keep-alive failed:", e)
        time.sleep(4)  # Keep under 5 seconds

threading.Thread(target=keep_alive, args=(client, UNIT_ID), daemon=True).start()

#write code that prints the BESS state when it changes
def print_bess_state(client, UNIT_ID):
    last_state = None
    while True:
        try:
            bess_state = client.read_input_registers(address=29, count=1, slave=UNIT_ID)
            if bess_state.isError():
                print(f"Error reading BESS state: {bess_state}")
            else:
                current_state = bess_state.registers[0]
                if current_state != last_state:
                    print(f"BESS State changed: {current_state}")
                    last_state = current_state
        except Exception as e:
            print("Error reading BESS state:", e)
        time.sleep(1)  # Check every 1 second

#make this a daemon thread so it doesn't block the main thread
threading.Thread(target=print_bess_state, args=(client, UNIT_ID), daemon=True).start()

typed = ""
while typed != "exit":
    typed = input("Type 'exit' to quit: ")
    
    if typed == "reset":
        if not client.is_socket_open():
            client.connect()
        response = client.write_coil(address=2, value=True, slave=UNIT_ID)
        if response.isError():
            print("Reset Write failed:", response)
        else:
            print("Reset Write successful.")
    elif typed == "stop":
        if not client.is_socket_open():
            client.connect()
        response = client.write_coil(address=1, value=True, slave=UNIT_ID)
        if response.isError():
            print("Stop Write failed:", response)
        else:
            print("Stop Write successful.")
    elif typed == "start":
        if not client.is_socket_open():
            client.connect()
        response = client.write_coil(address=0, value=True, slave=UNIT_ID)
        if response.isError():
            print("Start Write failed:", response)
        else:
            print("Start Write successful.")
    
    elif typed == "bess_state":
        if not client.is_socket_open():
            client.connect()
        bess_state = client.read_input_registers(address=29, count=1, slave=UNIT_ID)
        if bess_state.isError():
            print(f"Error reading BESS state: {bess_state}")
        else:
            print(f"BESS State: {bess_state.registers[0]}")
    
    elif typed == "bess_estop":
        if not client.is_socket_open():
            client.connect()
        bess_estop = client.read_input_registers(address=33, count=1, slave=UNIT_ID)
        if bess_estop.isError():
            print(f"Error reading BESS E-Stop: {bess_estop}")
        else:
            print(f"BESS E-Stop: {bess_estop.registers[0]}")
            #print in binary format
            print(f"BESS E-Stop (binary): {bess_estop.registers[0]:016b}")
    
    elif typed == "island":
        if not client.is_socket_open():
            client.connect()
        response = client.write_coil(address=4, value=False, slave=UNIT_ID)
        if response.isError():
            print("Islanding Write failed:", response)
        else:
            print("Islanding Write successful.")
    
    elif typed == "unisland":
        if not client.is_socket_open():
            client.connect()
        response = client.write_coil(address=4, value=True, slave=UNIT_ID)
        if response.isError():
            print("Unislanding Write failed:", response)
        else:
            print("Unislanding Write successful.")
    
    elif typed == "real_pow_0":
        if not client.is_socket_open():
            client.connect()
        response = client.write_register(address=2, value=0, slave=UNIT_ID)
        if response.isError():
            print("Real Power Write failed:", response)
        else:
            print("Real Power Write successful.")

    elif typed == "react_pow_0":
        if not client.is_socket_open():
            client.connect()
        response = client.write_register(address=3, value=0, slave=UNIT_ID)
        if response.isError():
            print("Reactive Power Write failed:", response)
        else:
            print("Reactive Power Write successful.")
    
    elif typed == "voltage":
        if not client.is_socket_open():
            client.connect()
        response = client.write_register(address=4, value=4800, slave=UNIT_ID)
        if response.isError():
            print("Voltage Write failed:", response)
        else:
            print("Voltage Write successful.")
    
    elif typed == "frequency":
        if not client.is_socket_open():
            client.connect()
        response = client.write_register(address=5, value=6000, slave=UNIT_ID)
        if response.isError():
            print("Frequency Write failed:", response)
        else:
            print("Frequency Write successful.")
    elif typed == "bess_state":
        if not client.is_socket_open():
            client.connect()
        bess_state = client.read_input_registers(address=76, count=1, slave=UNIT_ID)
        if bess_state.isError():
            print(f"Error reading BESS state: {bess_state}")
        else:
            print(f"BESS State: {bess_state.registers[0]}")
    #read input register 40 "Control Port"
    elif typed == "control_port":
        if not client.is_socket_open():
            client.connect()
        control_port = client.read_input_registers(address=40, count=1, slave=UNIT_ID)
        if control_port.isError():
            print(f"Error reading Control Port: {control_port}")
        else:
            print(f"Control Port: {control_port.registers[0]}")
    #read input register 41 "RH"
    elif typed == "rh":
        if not client.is_socket_open():
            client.connect()
        rh = client.read_input_registers(address=41, count=1, slave=UNIT_ID)
        if rh.isError():
            print(f"Error reading RH: {rh}")
        else:
            print(f"RH: {rh.registers[0]}")
    #read input register 30 "bat_sys_fault", it is a bitmap
    elif typed == "bat_sys_fault":
        if not client.is_socket_open():
            client.connect()
        bat_sys_fault = client.read_input_registers(address=30, count=1, slave=UNIT_ID)
        if bat_sys_fault.isError():
            print(f"Error reading Battery System Fault: {bat_sys_fault}")
        else:
            # Assuming the register is a bitmap, we can display it as a list of bits
            bits = [bool(bat_sys_fault.registers[0] & (1 << i)) for i in range(16)]
            print(f"Battery System Fault: {bits}")
    #write holding register 1 to 2 to enable the BESS + INV
    elif typed == "enable_bess_inv":
        if not client.is_socket_open():
            client.connect()
        response = client.write_register(address=1, value=2, slave=UNIT_ID)
        if response.isError():
            print("Enable BESS + INV Write failed:", response)
        else:
            print("Enable BESS + INV Write successful.")
    #get phase a current from input register 7
    elif typed == "phase_a_current":
        if not client.is_socket_open():
            client.connect()
        phase_a_current = client.read_input_registers(address=7, count=1, slave=UNIT_ID)
        if phase_a_current.isError():
            print(f"Error reading Phase A Current: {phase_a_current}")
        else:
            print(f"Phase A Current: {phase_a_current.registers[0]} A")
    elif typed == "EPC_state":
        if not client.is_socket_open():
            client.connect()
        epc_state = client.read_input_registers(address=15, count=1, slave=UNIT_ID)
        if epc_state.isError():
            print(f"Error reading EPC State: {epc_state}")
        else:
            print(f"EPC State: {epc_state.registers[0]}")
    #read islanding breaker state from coil 4
    elif typed == "islanding_breaker_state":
        if not client.is_socket_open():
            client.connect()
        islanding_breaker_state = client.read_coils(address=4, count=1, slave=UNIT_ID)
        if islanding_breaker_state.isError():
            print(f"Error reading Islanding Breaker State: {islanding_breaker_state}")
        else:
            print(f"Islanding Breaker State: {'Closed' if islanding_breaker_state.bits[0] else 'Open'}")
    #reconnect write 1 to coil 3
    elif typed == "reconnect":
        if not client.is_socket_open():
            client.connect()
        response = client.write_coil(address=3, value=True, slave=UNIT_ID)
        if response.isError():
            print("Reconnect Write failed:", response)
        else:
            print("Reconnect Write successful.")    

    #read bess_alm1 from holding register 31
    elif typed == "bess_alm1":
        if not client.is_socket_open():
            client.connect()
        bess_alm1 = client.read_input_registers(address=31, count=1, slave=UNIT_ID)
        if bess_alm1.isError():
            print(f"Error reading BESS Alarm 1: {bess_alm1}")
        else:
            print(f"BESS Alarm 1: {bess_alm1.registers[0]}")
            #print in binary format
            print(f"BESS Alarm 1 (binary): {bess_alm1.registers[0]:016b}")
    #read holding register 10 "inverter antiislanding"
    elif typed == "inverter_antiislanding":
        if not client.is_socket_open():
            client.connect()
        inverter_antiislanding = client.read_holding_registers(address=10, count=1, slave=UNIT_ID)
        if inverter_antiislanding.isError():
            print(f"Error reading Inverter Anti-Islanding: {inverter_antiislanding}")


        else:
            print(f"Inverter Anti-Islanding: {inverter_antiislanding.registers[0]}")
    #write 0 to coil 3 "estop_off"
    elif typed == "estop_off":
        if not client.is_socket_open():
            client.connect()
        response = client.write_coil(address=3, value=False, slave=UNIT_ID)
        if response.isError():
            print("E-Stop Off Write failed:", response)
        else:
            print("E-Stop Off Write successful.")
    #read coil 3 
    elif typed == "estop_read":
        if not client.is_socket_open():
            client.connect()
        response = client.read_coils(address=3, count=1, slave=UNIT_ID)
        if response.isError():
            print("E-Stop read failed:", response)
        else:
            print("E-Stop read successful.")
            print(f"E-Stop State: {'On' if response.bits[0] else 'Off'}")
    # read holding register 9 INV mode read
    elif typed == "inv_mode_read":
        if not client.is_socket_open():
            client.connect()
        response = client.read_holding_registers(address=9, count=1, slave=UNIT_ID)
        if response.isError():
            print("INV Mode read failed:", response)
        else:
            print("INV Mode read successful.")
            print(f"INV Mode: {response.registers[0]}")
    # command inv mode to 2
    elif typed == "inv_mode_following":
        if not client.is_socket_open():
            client.connect()
        response = client.write_register(address=9, value=2, slave=UNIT_ID)
        if response.isError():
            print("INV Mode command failed:", response)
        else:
            print("INV Mode command successful.")
    # command inv to mode to 1
    elif typed == "inv_mode_forming":
        if not client.is_socket_open():
            client.connect()
        response = client.write_register(address=9, value=1, slave=UNIT_ID)
        if response.isError():
            print("INV Mode command failed:", response)
        else:
            print("INV Mode command successful.")
    #command inv mode to 0 "none"
    elif typed == "inv_mode_none":
        if not client.is_socket_open():
            client.connect()
        response = client.write_register(address=9, value=0, slave=UNIT_ID)
        if response.isError():
            print("INV Mode command failed:", response)
        else:
            print("INV Mode command successful.")
    #command 0 to antiislanding register 10
    elif typed == "antiislanding_0":
        if not client.is_socket_open():
            client.connect()
        response = client.write_register(address=10, value=0, slave=UNIT_ID)
        if response.isError():
            print("Anti-Islanding command failed:", response)
        else:
            print("Anti-Islanding command successful.")
    #command 1 to antiislanding register 10
    elif typed == "antiislanding_1":
        if not client.is_socket_open():
            client.connect()
        response = client.write_register(address=10, value=1, slave=UNIT_ID)
        if response.isError():
            print("Anti-Islanding command failed:", response)
        else:
            print("Anti-Islanding command successful.")
    elif typed == "antiislanding_2":
        if not client.is_socket_open():
            client.connect()
        response = client.write_register(address=10, value=2, slave=UNIT_ID)
        if response.isError():
            print("Anti-Islanding command failed:", response)
        else:
            print("Anti-Islanding command successful.")

    #read power command register 2
    elif typed == "power_command_read":
        if not client.is_socket_open():
            client.connect()
        response = client.read_holding_registers(address=2, count=1, slave=UNIT_ID)
        if response.isError():
            print("Power Command read failed:", response)
        else:
            print("Power Command read successful.")
            print(f"Power Command: {response.registers[0]} kW")
    elif typed == "reactive_power_command_read":
        if not client.is_socket_open():
            client.connect()
        response = client.read_holding_registers(address=3, count=1, slave=UNIT_ID)
        if response.isError():
            print("Reactive Power Command read failed:", response)
        else:
            print("Reactive Power Command read successful.")
            print(f"Reactive Power Command: {response.registers[0]} kVAR")

    #read bess grid state from input register 38
    elif typed == "read_grid_state":
        if not client.is_socket_open():
            client.connect()
        response = client.read_input_registers(address=38, count=1, slave=UNIT_ID)
        if response.isError():
            print("BESS Grid State read failed:", response)
        else:
            print("BESS Grid State read successful.")
            print(f"BESS Grid State: {response.registers[0]}")

    #read coils 0 to 5
    elif typed == "read_coils_0_5":
        if not client.is_socket_open():
            client.connect()
        response = client.read_coils(address=0, count=6, slave=UNIT_ID)
        if response.isError():
            print("Coils read failed:", response)
        else:
            print("Coils read successful.")
            for i in range(6):
                print(f"Coil {i}: {'On' if response.bits[i] else 'Off'}")
    
    #read holding registers 0 to 10
    elif typed == "read_holding_registers_0_10":
        if not client.is_socket_open():
            client.connect()
        response = client.read_holding_registers(address=0, count=11, slave=UNIT_ID)
        if response.isError():
            print("Holding Registers read failed:", response)
        else:
            print("Holding Registers read successful.")
            for i in range(11):
                print(f"Register {i}: {response.registers[i]}")
    
    #read input registers 0 to 43
    elif typed == "read_input_registers_0_43":
        if not client.is_socket_open():
            client.connect()
        response = client.read_input_registers(address=0, count=44, slave=UNIT_ID)
        if response.isError():
            print("Input Registers read failed:", response)
        else:
            print("Input Registers read successful.")
            for i in range(44):
                print(f"Register {i}: {response.registers[i]}")
    #command power holding register 2, take command as input
    elif typed.startswith("power"):
        if not client.is_socket_open():
            client.connect()
        try:
            power_value = int(typed.split()[1])
            power_value = power_value * 10  # Convert kW to the expected register value
            response = client.write_register(address=2, value=power_value, slave=UNIT_ID)
            if response.isError():
                print("Power Command write failed:", response)
            else:
                print(f"Power Command write successful: {power_value/10} kW")
        except (IndexError, ValueError):
            print("Please provide a valid power value after 'power' command.")
    #command inverter to charge holding register 2, take command as input
    elif typed.startswith("charge"):
        if not client.is_socket_open():
            client.connect()
        try:
            charge_value = int(typed.split()[1])
            scaled_value = charge_value * 10  # Convert kW to register units (e.g., 10 kW → 100)

            # Handle negative values using two's complement encoding
            if not (-32768 <= scaled_value <= 32767):
                print("Charge value out of 16-bit signed range.")
            else:
                builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.BIG)
                builder.add_16bit_int(scaled_value)  # signed int, encodes correctly for 16-bit
                payload = builder.to_registers()
                response = client.write_registers(address=2, values=payload, slave=1)

                if response.isError():
                    print("Charge Command write failed:", response)
                else:
                    print(f"Charge Command write successful: {scaled_value / 10} kW")

        except (IndexError, ValueError):
            print("Please provide a valid charge value after 'charge' command.")
        #read power at Input register 3
    
    elif typed == "read_power":
        if not client.is_socket_open():
            client.connect()
        response = client.read_input_registers(address=3, count=1, slave=UNIT_ID)
        if response.isError():
            print("Power read failed:", response)
        else:
            print("Power read successful.")
            print(f"Power: {response.registers[0]} kW")

    #read reactive power at Input register 4
    elif typed == "read_reactive_power":
        if not client.is_socket_open():
            client.connect()
        response = client.read_input_registers(address=4, count=1, slave=UNIT_ID)
        if response.isError():
            print("Reactive Power read failed:", response)
        else:
            print("Reactive Power read successful.")
            print(f"Reactive Power: {response.registers[0]} kVAR")
    #read currents at input registers 7, 8, 9
    elif typed == "read_currents":
        if not client.is_socket_open():
            client.connect()
        response = client.read_input_registers(address=7, count=3, slave=UNIT_ID)
        if response.isError():
            print("Currents read failed:", response)
        else:
            print("Currents read successful.")
            print(f"Phase A Current: {response.registers[0]} A")
            print(f"Phase B Current: {response.registers[1]} A")
            print(f"Phase C Current: {response.registers[2]} A")
    #read SOC IR 18
    elif typed == "read_soc":
        if not client.is_socket_open():
            client.connect()
        response = client.read_input_registers(address=18, count=1, slave=UNIT_ID)
        if response.isError():
            print("SOC read failed:", response)
        else:
            print("SOC read successful.")
            print(f"SOC: {response.registers[0]} %")

    

"""
#test commands
client.write_coil(address=2, value=True, slave=1) # Reset the device if needed
time.sleep(1)  # Wait for reset to complete
bess_state = client.read_input_registers(address=29, count=1, slave=1) # Check BESS state (must return 1)
print(f"BESS State: {bess_state.registers[0]}")
if bess_state.registers[0] != 1:
    print("BESS is not in the expected state. Exiting.")
bess_estop = client.read_input_registers(address=33, count=1, slave=1)
print(f"BESS E-Stop: {bess_estop.registers[0]}")
if bess_estop.registers[0] != 65535:
    print("BESS is not in the expected E-Stop state. Exiting.")
    client.close()
    exit(1)
"""
"""
#commands
client.write_coil(address=2, value=True, unit=1) # Reset the device if needed
bess_state = client.read_input_registers(address=29, count=1, unit=1) # Check BESS state (must return 1)
if bess_state.registers[0] != 1:
    print("BESS is not in the expected state. Exiting.")
    client.close()
    exit(1)
bess_estop = client.read_input_registers(address=33, count=1, unit=1)
if bess_estop.registers[0] != 65535:
    print("BESS is not in the expected E-Stop state. Exiting.")
    client.close()
    exit(1)

# write 1 to coil 0 to start the BESS
client.write_coil(address=0, value=True, unit=1) # Start the BESS
# Read input register 29 to check the BESS state
bess_state = client.read_input_registers(address=29, count=1, unit=1)
while bess_state.registers[0] < 6:
    bess_state = client.read_input_registers(address=29, count=1, unit=1)
    if bess_state.registers[0] == 2:
        print("BESS is starting up.")  
    elif bess_state.registers[0] == 4:
        print("BESS is now on battery power.")
    elif bess_state.registers[0] == 6:
        print("BESS is running in following mode.")
    else: 
        print(f"BESS is in an unexpected state: {bess_state.registers[0]}")

# Set inverter output commands
client.write_register(address=2, value=2500, unit=1)  # Set Real Power (250kW)
client.write_register(address=3, value=0, unit=1)   # Set Reactive Power
client.write_register(address=4, value=4800, unit=1) # Set Voltage
client.write_register(address=5, value=6000, unit=1)  # Set Frequency``

if input("Press Enter to open the islanding breaker...") == "":
    pass
#just after breaker is opened:
client.write_coil(address=4, value=False, unit=1)  # Open the islanding breaker
# Read BESS state to confirm transition
bess_state = client.read_input_registers(address=29, count=1, unit=1)
if bess_state.registers[0] == 5:
    print("BESS has transitioned to running in forming mode.")

"""

if input("Press Enter to close connection") == "":
    pass
# Close connection
client.close()
