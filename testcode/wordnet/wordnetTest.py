def wnid_to_synset(wnid):
    from nltk.corpus import wordnet as wn  # entire script should not depend on wn

    offset = int(wnid[1:])
    pos = wnid[0]

    try:
        return wn.synset_from_pos_and_offset(wnid[0], offset)
    except:
        return FakeSynset(wnid)

def word_to_synset(word):
  from nltk.corpus import wordnet as wn  # entire script should not depend on wn
  return wn.synsets(word)

def synset_to_wnid(synset):
    return f'{synset.pos()}{synset.offset():08d}'

def wnid_to_name(wnid):
    return synset_to_name(wnid_to_synset(wnid))

def synset_to_name(synset):
    return synset.name().split('.')[0]

child_synsets = []
child_synsets.append(wnid_to_synset("n01503061"))
child_synsets.append(wnid_to_synset("n02121620"))
common_hypernyms = set(child_synsets[0].common_hypernyms(child_synsets[1]))
closest_common_hypernyms = set(child_synsets[0].lowest_common_hypernyms(child_synsets[1]))

print(child_synsets)
print(word_to_synset('bird'))
print(synset_to_wnid(word_to_synset('bird')[0]))
print(common_hypernyms)
print(closest_common_hypernyms)
