"""Exhaustively qualified replacement over a declared finite input domain."""
def verified_replacement(reference, candidate, *, max_cases=100000):
    a,b=reference.graph(),candidate.graph()
    if a['semantics']!='exact' or b['semantics']!='exact': raise ValueError('replacement requires exact semantics')
    if any(n['op'] in ('state','call','parameter') for n in a['nodes']+b['nodes']):
        raise ValueError('freeze parameters and remove state/host boundaries before replacement')
    def signature(ir): return sorted((n['name'],n['domain']['dtype'],n['domain']['min'],n['domain']['max']) for n in ir['nodes'] if n['op']=='input')
    if signature(a)!=signature(b) or set(a['outputs'])!=set(b['outputs']): raise ValueError('replacement signature mismatch')
    report=candidate.verify(lambda **inputs:reference.run(inputs),max_cases=max_cases)
    if not report['passed']: raise ValueError(f"replacement counterexample: {report['counterexample']}")
    return candidate
