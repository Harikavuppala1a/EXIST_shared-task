from ast import literal_eval

def load_map(filename):
    conf_sep = "----------"
    content = ''
    with open(filename, 'r') as f:
      for line in f:
        line = line.strip()
        if line != '' and line[0] != '#':
          content += line

    items = content.split(conf_sep)
    conf_map = {}
    for item in items:
      parts = [x.strip() for x in item.split('=')]
      conf_map[parts[0]] = literal_eval(parts[1])
    # print(conf_map)
    return conf_map
    
def load_config(filename):
  print("loading config")
  conf_sep_1 = "----------\n"
  conf_sep_2 = "**********\n"
  conf_dict_list = []
  conf_dict_com = {}
  with open(filename, 'r') as f:
    content = f.read()
  break_ind = content.find(conf_sep_2)  

  nested_comps = content[:break_ind].split(conf_sep_1)
  for comp in nested_comps:
    pairs = comp.split(';')
    conf_dict = {}
    for pair in pairs:
      pair = ''.join(pair.split())
      if pair == "" or pair[0] == '#': 
        continue
      parts = pair.split('=')
      conf_dict[parts[0]] = literal_eval(parts[1])
    conf_dict_list.append(conf_dict)

  lines = content[break_ind+len(conf_sep_2):].split('\n')
  for pair in lines:
    pair = ''.join(pair.split())
    if pair == "" or pair[0] == '#': 
      continue
    parts = pair.split('=')
    conf_dict_com[parts[0]] = literal_eval(parts[1])

  print("config loaded")
  return conf_dict_list, conf_dict_com