"""
OG RNNs would take the entire input and transform it into a single memory cell hidden state, thus memorizing the entire input of the sequence, so SINGLE HIDDEN STATE must remember the entire encoded input.

Problem is that the RNN can't directly access earlier hidden states from the encoder during the decoding phase, so it has to rely exclusively on the current hidden state which encapsulates all relevant information - easy to lose all context expecially in complex sentences where dependencies span long distances

This motivated the design of attention mechanisms. These allow the decoder to selectively access different parts of the sequence at each decoding step, thus can consider the relevancy or 'attend to' all other positions in the same sequence when considering the representation of a sequence.

"""

