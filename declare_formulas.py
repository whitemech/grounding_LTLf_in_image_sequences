formulas = []

formulas_names = []

# init(A)
formulas.append("c0")
formulas_names.append("Init(c0)")
#responded existence(A, B)
formulas.append("F c0 -> F c1")
formulas_names.append("Responded existence(c0,c1)")
#response(A,B)
formulas.append("G( c0 -> F c1)")
formulas_names.append("Response(c0,c1)")
#precedence(A,B)
formulas.append("(! c1 U c0) | G (! c1)")
formulas_names.append("Precedence(c0,c1)")
#succession(A,B)
formulas.append('G( c0 -> F c1) & (! c1 U c0) | G (! c1)')
formulas_names.append("Succession(c0,c1)")
#alternate response(A,B)
formulas.append('G( c0 -> X (! c0 U c1))')
formulas_names.append("Alternate response(c0,c1)")
#alternate precedence(A,B)'
formulas.append('(! c1 U c0) | G(! c1) & G (c1 -> X ((!c1 U c0) | G (!c1)))')
formulas_names.append("Alternate precedence(c0,c1)")
#alternate succession
formulas.append('G( c0 -> X (! c0 U c1)) & (! c1 U c0) | G (! c1)')
formulas_names.append("Alternate succesion(c0, c1)")
#chain response
formulas.append('G(c0 -> X c1)')
formulas_names.append("Chain response(c0, c1)")
#chain precedence
formulas.append('G((X c1) -> c0)')
formulas_names.append("Chain precedence(c0,c1)")
#not co-existence
formulas.append('!(F c0 & F c1)')
formulas_names.append("Not co-existence(c0,c1)")
#not succession
formulas.append('G (c0 -> ! (F c1))')
formulas_names.append("Not succession(c0,c1)")
#not chain succession
formulas.append('G(c0 -> ! (X c1))')
formulas_names.append("Not chain succession(c0,c1)")
#choice
formulas.append('F c0 | F c1')
formulas_names.append("Choice(c0,c1)")
#existence(c0,2)
formulas.append("F(c0 & X ( F(c0)))")
formulas_names.append("Existence(c0,2)")
#absence(c0,2)
formulas.append("!(F(c0 & X ( F(c0))))")
formulas_names.append("Absence(c0,2)")
#exactly(c0,2)
formulas.append("(F(c0 & X ( F(c0)))) & !(  F(c0 & X ( F ( F (c0 & X( F(c0)))))  )  )")
formulas_names.append("Exactly(c0,2)")
#co-existence(c0,c1)
formulas.append("(F(c1)) -> (F(c0))")
formulas_names.append("Co-existence(c0,c1)")
#chain succession(c0,c1)
formulas.append("G((c0 <-> X(c1)))")
formulas_names.append("Chain succession(c0,c1)")
#exclusive choice
formulas.append('(F c0 | F c1) & ! (F c0 & F c1)')
formulas_names.append("Exclusive choice(c0,c1)")
