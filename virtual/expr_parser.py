import math

def debug(d):
  # print(f'[{d["type"]}] --> {d}')
  return d

class ExprParser:
  def __init__(self, expr):
    self.expr_ = expr
    self.pos = 0
    self.mapping = dict()

  def curr_char(self):
    while self.pos != len(self.expr_) and self.expr_[self.pos] == ' ':
      self.pos += 1
    if self.pos == len(self.expr_):
      return None
    return self.expr_[self.pos]

  def curr_token(self):
    # TODO: Maybe store state to the `pos`.
    while self.pos != len(self.expr_) and self.expr_[self.pos] == ' ':
      self.pos += 1

    ahead_pos = self.pos
    while ahead_pos != len(self.expr_) and (self.expr_[ahead_pos] != ' ' and self.expr_[ahead_pos] != ',' and self.expr_[ahead_pos] != '(' and self.expr_[ahead_pos] != ')'):
      ahead_pos += 1
    init_pos = ahead_pos
    while ahead_pos != len(self.expr_) and self.expr_[ahead_pos] == ' ':
      ahead_pos += 1

    self.mapping[self.pos] = ahead_pos
    token = self.expr_[self.pos : init_pos]
    return token
  
  def curr_coef(self):
    # TODO: Maybe store state to the `pos`.
    ahead_pos = self.pos
    while ahead_pos != len(self.expr_) and (self.expr_[ahead_pos].isdigit() or self.expr_[ahead_pos] == '.'):
      ahead_pos += 1
    init_pos = ahead_pos
    while ahead_pos != len(self.expr_) and self.expr_[ahead_pos] == ' ':
      ahead_pos += 1

    self.mapping[self.pos] = ahead_pos
    coef = float(self.expr_[self.pos : init_pos])
    return coef

  def curr_literal(self):
    # TODO: Maybe store state to the `pos`.
    assert self.curr_char() == '"'
    ahead_pos = self.pos + 1
    while ahead_pos != len(self.expr_) and self.expr_[ahead_pos] != '"':
      ahead_pos += 1
    ahead_pos += 1
    init_pos = ahead_pos
    while ahead_pos != len(self.expr_) and self.expr_[ahead_pos] == ' ':
      ahead_pos += 1

    self.mapping[self.pos] = ahead_pos
    literal = self.expr_[self.pos : init_pos]
    return literal

  def advance_char(self):
    self.pos += 1

  def consume(self):
    assert self.pos in self.mapping
    self.pos = self.mapping[self.pos]
    self.mapping.pop(self.pos, None)

  def advance_token(self):
    self.consume()
  
  def advance_coef(self):
    self.consume()

  def advance_literal(self):
    self.consume()

  def parse(self):
    return self.E()

  def when_expr(self):
    # print(f'[when]')
    assert self.curr_token() == 'when'
    self.advance_token()
    cond = self.E()
    assert self.curr_token() == 'then'
    self.advance_token()
    then_stmt = self.E()
    return {
      'type' : 'when',
      'cond' : cond,
      'then' : then_stmt
    }

  def case_expr(self):
    # print(f'[case]')

    assert self.curr_token() == 'case'
    self.advance_token()
    assert self.curr_token() == 'when'

    when_stmts = []
    while self.curr_token() != 'else':
      when_stmts.append(self.when_expr())

    # Parse the end.
    self.advance_token()

    # Parse the `else`.
    else_stmt = self.E()

    assert self.curr_token() == 'end'
    self.advance_token()

    return debug({
      'type' : 'case',
      'when' : when_stmts,
      'else' : else_stmt
    })
  
  def round_expr(self):
    # print(f'[round] {self.expr_[self.pos:]}')
    assert self.curr_token() == 'round'
    self.advance_token()
    assert self.curr_char() == '('
    self.advance_char()

    expr = self.E()
    assert self.curr_char() == ','
    self.advance_char()
    scale = self.T()
    self.advance_char()
    return debug({
      'type' : 'round',
      'expr' : expr,
      'scale' : scale
    })

  def coalesce_expr(self):
    # print(f'[coalesce]')

    assert self.curr_token() == 'coalesce'
    self.advance_token()
    assert self.curr_char() == '('
    self.advance_char()

    exprs = []
    exprs.append(self.E())
    while self.curr_char() == ',':
      self.advance_char()
      exprs.append(self.E())
    self.advance_char()
    assert len(exprs) == 2
    return debug({
      'type' : 'coalesce',
      'exprs' : exprs
    })

  # Represents a generic term.
  def T(self):
    # print(f'[term] {self.expr_[self.pos:]}')

    if self.curr_char() == '(':
      self.advance_char()
      expr = self.E()
      self.advance_char()
      return debug({
        'type' : 'term',
        'coef' : 1.0,
        'expr' : expr
      })
    
    # CASE.
    if self.curr_token() == 'case':
      ret = self.case_expr()
      return ret
    
    # COALESCE.
    if self.curr_token() == 'coalesce':
      ret = self.coalesce_expr()
      return ret

    # ROUND.
    if self.curr_token() == 'round':
      ret = self.round_expr()
      return ret

    # NULL.
    if self.curr_token() == 'null':
      self.advance_token()
      return debug({
        'type' : 'literal',
        'val' : 'null'
      })

    # Literal.
    if self.curr_char() == '"':
      literal = self.curr_literal()

      self.advance_literal()
      # print(f'literal={literal}')
      return debug({
        'type' : 'literal',
        'val' : literal
      })

    # Unary minus.
    if self.curr_char() == '-':
      self.advance_char()
      expr = self.T()
      return debug({
        'type' : 'term',
        'coef' : -1.0,
        'expr' : expr
      })
    
    # Parse coefficient.
    if self.curr_char().isdigit():
      coef = self.curr_coef()
      self.advance_coef()

    # Do we have a multiplication?
    if self.curr_char() == '*':
      self.advance_char()
      expr = self.T()
      return debug({
        'type' : 'term',
        'coef' : coef,
        'expr' : expr
      })

    # Otherwise, we have a simple numeric. This is for the intercept or for the round scale.
    return debug({
      'type' : 'literal',
      'val' : coef
    })

  # Represents a `CAST`.
  def C(self):
    # print(f'[cast]')

    expr = self.T()
    if self.curr_char() == ':':
      self.advance_char()
      assert self.curr_char() == ':'
      self.advance_char()
      cast_type = self.curr_token()
      self.advance_token()
      return debug({
        'type' : 'cast',
        'expr' : expr,
        'cast_type' : cast_type
      })
    return expr

  # Represents a generic expression.
  def E(self):
    # print('[expr]')

    # Generic case.
    exprs = [self.C()]
    while self.curr_char() == '+':
      self.advance_char()
      exprs.append(self.C())

    # If we anyway have a single expression, just take that as the root.
    if len(exprs) == 1:
      return exprs[0]
    
    # Otherwise, build a sum.
    return {
      'type' : 'sum',
      'terms' : exprs
    }


def print_expr(tree):
  if tree['type'] == 'cast':
    expr = print_expr(tree['expr'])
    assert expr
    if not expr[-1] in [')', '"']:
      expr = f'({expr})'
    return f"{expr}::{tree['cast_type']}"

  if tree['type'] == 'case':
    whens = '\n'.join(map(print_expr, tree['when']))
    else_stmt = print_expr(tree['else'])
    return f'(case {whens} else {else_stmt} end)'

  if tree['type'] == 'when':
    cond = print_expr(tree['cond'])
    then_stmt = print_expr(tree['then'])
    return f'when {cond} then {then_stmt}'
  
  if tree['type'] == 'coalesce':
    exprs = ', '.join(map(print_expr, tree['exprs']))
    return f'coalesce ({exprs})'
  
  if tree['type'] == 'round':
    return f"round({print_expr(tree['expr'])}, {print_expr(tree['scale'])})"

  if tree['type'] == 'sum':
    exprs = ' + '.join(map(print_expr, tree['terms']))
    return exprs
  
  if tree['type'] == 'or':
    factors = ' or '.join(map(print_expr, tree['factors']))
    return factors

  if tree['type'] == 'term':
    coef = tree['coef']

    # Optimize out an 1.
    if math.isclose(coef, 1.0):
      return f"{print_expr(tree['expr'])}"
    
    # Optimize out an -1.
    if math.isclose(coef, -1.0):
      return f"-{print_expr(tree['expr'])}"
    
    # Cast to an integer.
    if coef.is_integer():
      coef = int(coef)

    # Generic expresison.
    return f"{coef} * {print_expr(tree['expr'])}"

  if tree['type'] == 'literal':
    if isinstance(tree['val'], str):
      if tree['val'].isdigit():
        tmp = float(tree['val'])
        if tmp.is_integer():
          return f"{int(tmp)}"
    else:
      if tree['val'].is_integer():
        return f"{int(tree['val'])}"
    return f"{tree['val']}"
  
  assert 0  
