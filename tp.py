import lgpio
import time

LED = 17
print(lgpio.gpio_get_chip_info)
h = lgpio.gpiochip_open(0)
lgpio.gpio_claim_output(h, LED)

while True:
    try:
        lgpio.gpio_write(h,LED, 1)
        time.sleep(1)
        lgpio.gpio_write(h,LED, 0)
        time.sleep(1)
    except KeyboardInterrupt:
        lgpio.gpiochip_close(h)