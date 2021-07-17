# %%
import operator

# s = sorted(s, key = operator.itemgetter(1, 2))

matches = [
    (15615209398620108843, 113, 114),  # a
    (5408755460585001224, 125, 127),  # d
    (5408755460585001224, 113, 115),  # b
    (5408755460585001224, 114, 116),  # c
]


def match_overlap(matches):
    sorted_matches = sorted(matches, key=operator.itemgetter(1, 2))
    match_ranges = [range(match[1], match[2]) for match in sorted_matches]
    indeces_to_remove = set()
    new_matches = []
    for index, elem in enumerate(match_ranges):
        if index < (len(match_ranges) - 1):
            this_elem = set(elem)
            next_elem = match_ranges[index + 1]
            if intersect := this_elem.intersection(next_elem):
                new_matches.append((f"new_match_{index}", elem[0], next_elem[-1] + 1))
                indeces_to_remove.add(index)
                indeces_to_remove.add(index + 1)

    if not indeces_to_remove:
        return sorted_matches

    sorted_matches_dict = {index: value for index, value in enumerate(sorted_matches)}

    for index in indeces_to_remove:
        sorted_matches_dict.pop(index, None)

    sorted_matches = list(sorted_matches_dict.values())

    sorted_matches = sorted(sorted_matches + new_matches, key=operator.itemgetter(1, 2))

    sorted_matches = match_overlap(sorted_matches)

    return sorted_matches


print(match_overlap(matches))

# %%
