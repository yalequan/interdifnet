# InterDIFNet Package Dependencies

## Interactive Package Management

The `InterDIFNet.py` module includes **interactive package management**. When you import from InterDIFNet, the module will:

1. **Check** if all required packages are installed.
2. **List** any missing packages with installation instructions.
3. **Prompt you** to choose automatic installation or manual installation.
4. **Respect your choice** - no forced downloads.

## How It Works

When you run:
```python
from InterDIFNet import Simulation_Study, train_InterDIFNet, DIF_Detection
```

### If all packages are installed:
```
All InterDIFNet dependencies are installed.
```

### If packages are missing:
```
======================================================================
InterDIFNet Dependencies Check
======================================================================

The following packages are required but not installed:
  - tensorflow
  - scikit-multilearn

You can install them by running:
  pip install tensorflow scikit-multilearn

Would you like to install them automatically now? (y/n):
```

You can then:
- Type **`y`** or **`yes`** to install automatically
- Type **`n`** or **`no`** to exit and install manually
- Press **Ctrl+C** to cancel

## Required Packages

The following packages are automatically managed:

| Package | PyPI Name | Import Name |
|---------|-----------|-------------|
| NumPy | `numpy` | `numpy` |
| Pandas | `pandas` | `pandas` |
| TensorFlow | `tensorflow` | `tensorflow` |
| Scikit-learn | `scikit-learn` | `sklearn` |
| Matplotlib | `matplotlib` | `matplotlib` |
| Seaborn | `seaborn` | `seaborn` |
| Scikit-multilearn | `scikit-multilearn` | `skmultilearn` |
| NetworkX | `networkx` | `networkx` |

## Output Messages

### All Dependencies Installed:
```
All InterDIFNet dependencies are installed.
```

### Missing Dependencies - Interactive Prompt:
```
======================================================================
InterDIFNet Dependencies Check
======================================================================

The following packages are required but not installed:
  - tensorflow
  - scikit-multilearn

You can install them by running:
  pip install tensorflow scikit-multilearn

Would you like to install them automatically now? (y/n): y

Installing packages...
  Installing 'tensorflow'...
  Successfully installed 'tensorflow'
  Installing 'scikit-multilearn'...
  Successfully installed 'scikit-multilearn'

All dependencies installed successfully!
```

### If You Choose Manual Installation:
```
Would you like to install them automatically now? (y/n): n

Please install the required packages before using InterDIFNet.
Run: pip install tensorflow scikit-multilearn
```

## Manual Installation

If you prefer to install packages manually, you can do so before importing InterDIFNet:

```bash
pip install numpy pandas tensorflow scikit-learn matplotlib seaborn scikit-multilearn networkx
```

## Troubleshooting

If automatic installation fails:
1. Check your internet connection
2. Ensure pip is up to date: `pip install --upgrade pip`
3. Manually install the failing package: `pip install <package-name>`
4. Check for permission issues (may need `sudo` on some systems or use `--user` flag)

## User Control

**You are always in control:**
- The module **never installs without asking**
- You can choose to install automatically or manually
- Clear instructions are provided for manual installation
- You can cancel at any time with Ctrl+C

## Automatic Installation in Scripts

If you want to use InterDIFNet in automated scripts where user input isn't possible, install all dependencies first:

```bash
pip install numpy pandas tensorflow scikit-learn matplotlib seaborn scikit-multilearn networkx
```

Then your script will import without prompts.
