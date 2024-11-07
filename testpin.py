import gpiod
import time

LED_PIN = 17
chip = gpiod.Chip('gpiochip0')
led_line = chip.get_line(LED_PIN)
led_line.request(consumer='LED', type=gpiod.LINE_REQ_DIR_OUT)
while True:
    led_line.set_value(1)