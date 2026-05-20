import re

with open("generate_deck.py", "r") as f:
    code = f.read()

def increase_pt(match):
    val = float(match.group(1))
    # Heuristics for increasing font sizes
    if val <= 10:
        new_val = val + 2
    elif val <= 14:
        new_val = val + 2.5
    elif val <= 20:
        new_val = val + 3
    elif val <= 30:
        new_val = val + 4
    else:
        new_val = val + 6
        
    if new_val.is_integer():
        return f"Pt({int(new_val)})"
    return f"Pt({new_val})"

new_code = re.sub(r'Pt\(([\d\.]+)\)', increase_pt, code)

with open("generate_deck.py", "w") as f:
    f.write(new_code)

print("Font sizes increased successfully.")
