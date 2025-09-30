# GPIO Wiring Guide - PiEdge EduKit

## LED Circuit (BCM17)

### Components Required

- Raspberry Pi (any model with GPIO)
- LED (any color, 3.3V compatible)
- Resistor (220Ω recommended)
- Breadboard and jumper wires

### Wiring Diagram

```
Pi GPIO Pin 11 (BCM17) → Resistor (220Ω) → LED (+) → LED (-) → Pi GND
```

### Pin Layout

- **BCM17** (Physical pin 11): LED control signal
- **GND** (Physical pin 6, 9, 14, 20, 25, 30, 34, 39): Ground

### Circuit Details

- LED forward voltage: ~2V (typical)
- Resistor value: 220Ω (limits current to ~6mA)
- Pi GPIO voltage: 3.3V
- Current calculation: (3.3V - 2V) / 220Ω ≈ 6mA

### Safety Notes

- Always use a current-limiting resistor
- Never connect LED directly to GPIO pin
- Maximum GPIO current: 16mA per pin
- Total GPIO current: 50mA for all pins combined

## Testing Circuit

### Manual Test

```bash
# Test GPIO manually (requires GPIO permissions)
echo 17 > /sys/class/gpio/export
echo out > /sys/class/gpio/gpio17/direction
echo 1 > /sys/class/gpio/gpio17/value  # LED ON
echo 0 > /sys/class/gpio/gpio17/value  # LED OFF
echo 17 > /sys/class/gpio/unexport
```

### Software Test

```bash
# Test with PiEdge EduKit (simulation mode)
python -m piedge_edukit.gpio_control --simulate

# Test with real GPIO
python -m piedge_edukit.gpio_control
```

## Troubleshooting

### LED Not Lighting

1. Check wiring connections
2. Verify resistor value (220Ω)
3. Test LED with multimeter
4. Check GPIO permissions: `groups $USER`

### Permission Denied

```bash
sudo usermod -aG gpio $USER
# Logout and login again
```

### GPIO Not Available

- Use `--simulate` flag for testing without hardware
- Check if running on actual Raspberry Pi
- Verify 64-bit OS installation

