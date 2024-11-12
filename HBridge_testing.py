import lgpio
import time

# Open a handle to the GPIO chip (usually 0 for /dev/gpiochip0)
h = lgpio.gpiochip_open(0)

# Define GPIO pins (using BCM numbering)
INPUT_PINS = [2, 12]
OUTPUT_PINS = [3, 9, 10, 11, 13]

# Configure input pins (Note: lgpio may not support internal pull-ups directly)
for pin in INPUT_PINS:
    lgpio.gpio_claim_input(h, pin)

# Configure output pins and set initial state to LOW
for pin in OUTPUT_PINS:
    lgpio.gpio_claim_output(h, pin)
    lgpio.gpio_write(h, pin, 0)

# Function to emulate analogWrite for PWM
def analog_write_pwm(h, pin, duty_cycle):
    # lgpio provides hardware PWM support
    frequency = 1000  # 1 kHz frequency
    lgpio.tx_pwm(h, pin, frequency, duty_cycle / 255 * 100)  # Convert to percentage

# Start PWM on pin 11 at full duty cycle (255 out of 255)
analog_write_pwm(h, 11, 255)

try:
    while True:
        # Set initial states for outputs
        lgpio.gpio_write(h, 10, 0)
        lgpio.gpio_write(h, 9, 0)
        lgpio.gpio_write(h, 3, 0)
        lgpio.gpio_write(h, 13, 0)
        analog_write_pwm(h, 11, 255)  # Maintain PWM at full duty cycle

        # Read input pins
        input2 = lgpio.gpio_read(h, 2)
        input12 = lgpio.gpio_read(h, 12)

        # Check inputs and set outputs accordingly
        if input2 == 0:  # Active low input
            lgpio.gpio_write(h, 10, 1)
            lgpio.gpio_write(h, 9, 0)
            lgpio.gpio_write(h, 3, 1)
        elif input12 == 0:
            lgpio.gpio_write(h, 10, 0)
            lgpio.gpio_write(h, 9, 1)
            lgpio.gpio_write(h, 13, 1)

        time.sleep(0.01)  # Delay for 10 milliseconds

except KeyboardInterrupt:
    # Cleanup on exit
    for pin in OUTPUT_PINS:
        lgpio.gpio_write(h, pin, 0)  # Set all outputs to LOW
    lgpio.gpiochip_close(h)
    print("\nProgram terminated and GPIO cleaned up.")