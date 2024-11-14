import lgpio
import time

# Open a handle to the GPIO chip (usually 0 for /dev/gpiochip0)
h = lgpio.gpiochip_open(0)

# Define GPIO pins (using BCM numbering as per the Raspberry Pi 5 pinout)
INPUT_PINS = [2, 17]  # Example input pins, modify as needed
OUTPUT_PINS = [3, 10, 11, 13, 19]  # Example output pins, modify as needed

# Configure input pins
for pin in INPUT_PINS:
    lgpio.gpio_claim_input(h, pin)

# Configure output pins and set initial state to LOW
for pin in OUTPUT_PINS:
    lgpio.gpio_claim_output(h, pin)
    lgpio.gpio_write(h, pin, 0)

# Function to emulate analogWrite for PWM
def analog_write_pwm(h, pin, duty_cycle):
    frequency = 1000  # 1 kHz frequency (adjust as needed)
    lgpio.tx_pwm(h, pin, frequency, duty_cycle / 255 * 100)  # Convert duty cycle to percentage

# Start PWM on pin 19 (or any valid PWM pin) at full duty cycle (255 out of 255)
analog_write_pwm(h, 19, 255)

try:
    while True:
        # Set initial states for outputs
        lgpio.gpio_write(h, 11, 0)
        lgpio.gpio_write(h, 10, 0)
        lgpio.gpio_write(h, 3, 0)
        lgpio.gpio_write(h, 13, 0)
        analog_write_pwm(h, 19, 255)  # Maintain PWM at full duty cycle

        # Read input pins
        input2 = lgpio.gpio_read(h, 2)
        input17 = lgpio.gpio_read(h, 17)

        # Check inputs and set outputs accordingly
        if input2 == 0:  # Active low input
            lgpio.gpio_write(h, 11, 1)
            lgpio.gpio_write(h, 10, 0)
            lgpio.gpio_write(h, 3, 1)
        elif input17 == 0:
            lgpio.gpio_write(h, 11, 0)
            lgpio.gpio_write(h, 10, 1)
            lgpio.gpio_write(h, 13, 1)

        time.sleep(0.01)  # Delay for 10 milliseconds

except KeyboardInterrupt:
    # Cleanup on exit
    for pin in OUTPUT_PINS:
        lgpio.gpio_write(h, pin, 0)  # Set all outputs to LOW
    lgpio.gpiochip_close(h)
    print("\nProgram terminated and GPIO cleaned up.")
