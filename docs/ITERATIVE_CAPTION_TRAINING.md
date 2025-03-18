# Iterative Caption Training

Caption iteratively for training

1. Caption naively

2. Ask to improve the caption

- iteratively ask to improve the last caption after training
  - can ask for a new captain (no grad)
  - save new caption
- train on the new caption

3. Train # amount

- repeat 2. 3.

---

## Ideas

Keep track of how many we modified
Mark areas that should be something specific (masked loss marking)
