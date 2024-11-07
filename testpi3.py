import RPi.GPIO as GPIO
import time

LED_PIN =17
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN,GPIO.OUT)
try:
    while True:
        GPIO.output(LED_PIN, GPIO.HIGH)
except KeyboardInterrupt:
    pass
finally:
    GPIO.cleanup()