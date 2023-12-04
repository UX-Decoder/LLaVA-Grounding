def decide_succ(data):
    if 'gd_ls' not in data:
        return False
    gd_ls=data['gd_ls']
    if len(gd_ls)==0:
        return False
    gd_ls=[0 if gd is None else 1 for gd in gd_ls]
    if sum(gd_ls)==0:
        return False
    conv=data['conversations']
    question=conv[0]['value']
    if 'with grounding' not in question:
        return False
    answer=conv[1]['value']
    if 'Please provide' in answer:
        return False
    if answer.count('<g_s>')!=len(gd_ls):
        return False
    if answer.count('<g_e>')!=len(gd_ls):
        return False
    if answer.count('<seg>')!=len(gd_ls):
        return False
    return True