# Fuzzy Sets (as simple dictionaries with elements and their membership values)
A = {'x': 0.6, 'y': 0.8}
B = {'x': 0.4, 'y': 0.5}

# 1. Union of two fuzzy sets (max value for each element)
union = {}
for key in A:
    union[key] = max(A[key], B[key])

print("Union of A and B:", union)

# 2. Intersection of two fuzzy sets (min value for each element)
intersection = {}
for key in A:
    intersection[key] = min(A[key], B[key])

print("Intersection of A and B:", intersection)

# 3. Complement of a fuzzy set (1 - value for each element)
complement_A = {}
for key in A:
    complement_A[key] = 1 - A[key]

print("Complement of A:", complement_A)

# 4. Difference of two fuzzy sets (min of A and (1 - B) for each element)
difference = {}
for key in A:
    difference[key] = min(A[key], 1 - B[key])

print("Difference of A and B:", difference)

# --- Fuzzy Relations ---

# 5. Cartesian product of fuzzy sets A and B (combining each pair of elements from A and B)
R1 = {}
for a in A:
    for b in B:
        R1[(a, b)] = min(A[a], B[b])

print("Cartesian Product of A and B (R1):", R1)

# 6. Max-Min composition of two fuzzy relations (R1 and R2)
# Creating another fuzzy relation R2 from B and A
R2 = {}
for b in B:
    for a in A:
        R2[(b, a)] = min(B[b], A[a])

print("Cartesian Product of B and A (R2):", R2)

# Max-Min composition: combining R1 and R2
composition = {}
for (a, b) in R1:
    for (b2, c) in R2:
        if b == b2:  # Matching common element (b)
            key = (a, c)
            composition[key] = max(composition.get(key, 0), min(R1[(a, b)], R2[(b2, c)]))

print("Max-Min Composition of R1 and R2:", composition)
