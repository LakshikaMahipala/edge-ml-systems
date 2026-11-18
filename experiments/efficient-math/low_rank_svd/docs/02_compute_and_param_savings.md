Compute and parameter savings

Original W: (in x out)
Params: in*out
MACs per inference: in*out

Low rank r:
A: (in x r)
B: (r x out)
Params: in*r + r*out = r*(in+out)
MACs: same

Savings condition
r*(in+out) < in*out  => r < (in*out)/(in+out)

Example
in=1024, out=1024:
threshold r < (1024*1024)/(2048)=512
So r=128 is a big win; r=800 is not.
