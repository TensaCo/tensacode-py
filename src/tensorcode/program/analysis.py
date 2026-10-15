"""Conservative interval diagnostics; these are not equivalence proofs."""
import math
import struct


def analyze(program):
    ir=program.graph(); ranges=[]; warnings=[]
    for n in ir['nodes']:
        op=n['op']; args=[ranges[a] for a in n['args']]; d=n['domain']
        lo,hi=d['min'],d['max']; integer=d['dtype'] in ('int32','bool')
        if op in ('const','parameter','state') and op=='const': lo=hi=n['value']; integer=float(lo).is_integer()
        elif op in ('add','sub','mul','div'):
            a,b=args; integer=a['integer'] and b['integer'] and op!='div'
            if op=='add': lo,hi=a['min']+b['min'],a['max']+b['max']
            elif op=='sub': lo,hi=a['min']-b['max'],a['max']-b['min']
            elif op=='mul':
                products=[x*y for x in (a['min'],a['max']) for y in (b['min'],b['max'])]; lo,hi=min(products),max(products)
            elif b['min']<=0<=b['max']: warnings.append({'node':n['id'],'reason':'divisor may be zero'})
            else:
                values=[x/y for x in (a['min'],a['max']) for y in (b['min'],b['max'])];lo,hi=min(values),max(values)
            if not math.isfinite(lo) or not math.isfinite(hi) or lo<d['min'] or hi>d['max']:
                warnings.append({'node':n['id'],'reason':'arithmetic may exceed finite domain'})
            if integer and (lo < -9007199254740991 or hi > 9007199254740991):
                warnings.append({'node':n['id'],'reason':'integer arithmetic may lose precision'})
            # Preserve exact integer intervals; otherwise round outward.
            if not integer:
                lo=math.nextafter(lo,-math.inf); hi=math.nextafter(hi,math.inf)
        elif op in ('gt','ge','lt','le','eq','ne'): lo,hi=0,1;integer=(n.get('semantics',ir['semantics'])!='relaxed')
        elif op=='select': lo=min(args[1]['min'],args[2]['min']);hi=max(args[1]['max'],args[2]['max']);integer=args[1]['integer'] and args[2]['integer'] and n.get('semantics',ir['semantics'])!='relaxed'
        elif op=='gather':
            lo=min(a['min'] for a in args[1:]);hi=max(a['max'] for a in args[1:]);integer=all(a['integer'] for a in args[1:])
            if not args[0]['integer'] or args[0]['min']<0 or args[0]['max']>=len(args)-1: warnings.append({'node':n['id'],'reason':'gather index may be invalid'})
        elif op=='relu': lo=max(0,args[0]['min']);hi=max(0,args[0]['max']);integer=args[0]['integer']
        elif op=='neg': lo=-args[0]['max'];hi=-args[0]['min'];integer=args[0]['integer']
        elif op=='sigmoid':lo,hi=0,1
        elif op=='call':warnings.append({'node':n['id'],'reason':'explicit external computation boundary'})
        if d['dtype']=='float32':
            def round32(x):
                try: return struct.unpack('<f',struct.pack('<f',x))[0]
                except OverflowError: return math.copysign(math.inf,x)
            lo,hi=round32(lo),round32(hi)
        ranges.append({'min':max(d['min'],lo),'max':min(d['max'],hi),'integer':integer})
    return {'ranges':ranges,'warnings':warnings,'nodeCount':len(ir['nodes']), 'depth':len(program.schedule())}
