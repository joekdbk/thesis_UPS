from pymodbus.client import ModbusTcpClient
from pymodbus.payload import BinaryPayloadDecoder
from pymodbus.constants import Endian

# Device parameters
IP_ADDRESS = '172.31.212.16'
PORT = 502
UNIT_ID = 255

# Create Modbus client and connect
client = ModbusTcpClient(IP_ADDRESS, port=PORT)
if not client.connect():
    print("Failed to connect to device.")
    exit(1)

connection = client.connect()
#^gucci dont change this

if connection:
    # Modbus register addresses are 0-based
    # So register 1845 becomes 1844 in the function call
    start_address = 1844
    

# Helper to read a float from 2 input registers
def read_float32_input(register, label):
    response = client.read_holding_registers(address=register, count=2, slave=UNIT_ID)
    if response.isError():
        print(f"Error reading {label} at register {register}: {response}")
        return None
    decoder = BinaryPayloadDecoder.fromRegisters(
        response.registers, byteorder=Endian.BIG, wordorder=Endian.BIG
    )
    value = decoder.decode_32bit_float()
    return value

# Read phase currents (FLOAT32 = 2 registers each)
current_a = read_float32_input(2999, "Current A")
current_b = read_float32_input(3001, "Current B")
current_c = read_float32_input(3003, "Current C")

# Display the values
print(f"Current A: {current_a} A")
print(f"Current B: {current_b} A")
print(f"Current C: {current_c} A")

year = client.read_holding_registers(address=1836, count=1, slave=UNIT_ID)
print(f"Year: {year.registers[0]}")

month = client.read_holding_registers(address=1837, count=1, slave=UNIT_ID)
print(f"month: {month.registers[0]}")

day = client.read_holding_registers(address=1838, count=1, slave=UNIT_ID)
print(f"day: {day.registers[0]}")

hour = client.read_holding_registers(address=1839, count=1, slave=UNIT_ID)
print(f"hour: {hour.registers[0]}")

app_pow = read_float32_input(3075, "App Pow")
print(f"App Pow: {app_pow} V")

voltA_B = read_float32_input(3019, "Voltage A-B")
print(f"Voltage A-B2: {voltA_B} V")

#power check
calc_pow = current_a * voltA_B * 1.71
print(f"Calculated Power: {calc_pow} W")

# Close connection
client.close()
