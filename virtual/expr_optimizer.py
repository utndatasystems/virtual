def optimize(tree):
  # print(tree)

  if tree['type'] == 'cast':
    tree['expr'] = optimize(tree['expr'])
    return tree

  if tree['type'] == 'case':
    tree['when'] = list(map(optimize, tree['when']))
    tree['else'] = optimize(tree['else'])
    return tree

  if tree['type'] == 'when':
    tree['cond'] = optimize(tree['cond'])
    tree['then'] = optimize(tree['then'])
    return tree

  if tree['type'] == 'coalesce':
    tree['exprs'] = list(map(optimize, tree['exprs']))
    return tree

  if tree['type'] == 'round':
    tree['expr'] = optimize(tree['expr'])
    return tree

  if tree['type'] == 'sum':
    # First optimize the terms.
    tree['terms'] = list(map(optimize, tree['terms']))

    # Collect the local when-conditions.
    local_when_conds = []
    for term in tree['terms']:
      if term['type'] != 'term':
        continue
      if term['expr']['type'] == 'case':
        assert len(term['expr']['when']) == 1
        local_when_conds.append(term['expr']['when'][0]['cond'])

    # Nothing to optimize?
    if not local_when_conds:
      return tree

    # Merge the local when conditions.
    merged_when_conds = {
      'type' : 'or',
      'factors' : local_when_conds
    }

    # Collect the updated terms.
    updated_terms = []
    for term in tree['terms']:
      if term['type'] != 'term':
        updated_terms.append(term)
        continue
      if term['expr']['type'] != 'case':
        updated_terms.append(term)
        continue

      # Special case.
      # NOTE: We also optimize to be sure the coefficients are propagated, if any.
      # NOTE: Otherwise, we could also do this by hand, since there are at most 2 levels.
      new_stmt = optimize({
        'type' : 'term',
        'coef' : term['coef'],
        'expr' : term['expr']['else']
      })
      updated_terms.append(new_stmt)

    # TODO: Optimize the remaining terms.

    new_tree = {
      'type' : 'case',
      'when' : [{
        'type' : 'when',
        'cond' : merged_when_conds,
        'then' : {'type': 'literal', 'val': 'null'}
      }],
      'else' : {
        'type' : 'sum',
        'terms' : updated_terms
      }
    }

    return new_tree

  if tree['type'] == 'term':
    tree['expr'] = optimize(tree['expr'])
    if tree['expr']['type'] == 'term':
      assert 'coef' in tree['expr']
      merged_coef = tree['coef'] * tree['expr']['coef']
      return {
        'type' : 'term',
        'coef' : merged_coef,
        'expr' : tree['expr']['expr']
      }
    return tree

  if tree['type'] == 'literal':
    return tree

  assert 0