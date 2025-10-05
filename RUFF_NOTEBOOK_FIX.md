# Ruff Notebook Configuration

## What Was Added

The `.ruff.toml` now ignores these common notebook patterns that Ruff flags but are normal in Jupyter:

### New Ignores Added:
- **F821** - undefined name (defined in another cell)
- **E701** - multiple statements on one line (colon)
- **E702** - multiple statements on one line (semicolon)  
- **E722** - bare except (ok in demo cells)

### Complete Notebook Ignore List:
- **E501** - long lines in cells/markdown
- **I001** - import order
- **F401** - imported but unused (used in another cell)
- **F811** - redefinition between cells
- **F841** - assigned but never used
- **F821** - undefined name (defined in another cell)
- **E701** - multiple statements on one line (colon)
- **E702** - multiple statements on one line (semicolon)
- **F541** - f-string without placeholders
- **E722** - bare except (ok in demo cells)
- **UP015** - cosmetic

## Result
This should eliminate the remaining 36 Ruff warnings in notebooks while keeping strict linting for `src/` and `scripts/`.

## To Apply
1. **Ctrl+Shift+P** → **Developer: Reload Window**
2. **Ctrl+Shift+P** → **Ruff: Restart Server**

The Problems panel should now be clean!
