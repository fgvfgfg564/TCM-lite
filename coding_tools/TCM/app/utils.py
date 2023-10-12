def combine_bytes(bytestring1, bytestring2):
    combined_bytestring = len(bytestring1).to_bytes(4, byteorder='big') + bytestring1 + bytestring2
    return combined_bytestring

def separate_bytes(combined_bytestring):
    # Separating them
    size1 = int.from_bytes(combined_bytestring[:4], byteorder='big')

    bytestring1 = combined_bytestring[4:4+size1]
    bytestring2 = combined_bytestring[4+size1:]

    return bytestring1, bytestring2