import re
def find_centigrade(context):
  centigrade_syms=[
   r"\b12[0-1]\s?EC\b",
   r"\b1[01][0-9]\s?EC\b",
   r"\b[0-9][0-9]\s?EC\b",
   r"\b[0-9]\s?EC\b",
   r"\b12[0-1]\s?C\b",
   r"\b1[01][0-9]\s?C\b",
   r"\b[0-9][0-9]\s?C\b",
   r"\b[0-9]\s?C\b",
   r"\b12[0-1]\s?oC\b",
   r"\b1[01][0-9]\s?oC\b",
   r"\b[0-9][0-9]\s?oC\b",
   r"\b[0-9]\s?oC\b",
   r"\b12[0-1]\s?degrees\b",
   r"\b1[01][0-9]\s?degrees\b",
   r"\b[0-9][0-9]\s?degrees\b",
   r"\b[0-9]\s?degrees\b",
   r"\b12[0-1]\s?[(]deg[)]C\b",
   r"\b1[01][0-9]\s?[(]deg[)]C\b",
   r"\b[0-9][0-9]\s?[(]deg[)]C\b",
   r"\b[0-9]\s?[(]deg[)]C\b",
   r"\b12[0-1]\s?°C\b",
   r"\b1[01][0-9]\s?°C\b",
   r"\b[0-9][0-9]\s?°C\b",
   r"\b[0-9]\s?°C\b",
  ]
  for sym in centigrade_syms:
    if len(re.findall(sym,context))>0:
      return True
  
  return context.find("°C")!=-1

def find_name(seq,names):
  flag = 1
  for one_name_in_context in names:
    if seq.find(one_name_in_context)!=-1:
      flag = 0
      break
  return flag==0