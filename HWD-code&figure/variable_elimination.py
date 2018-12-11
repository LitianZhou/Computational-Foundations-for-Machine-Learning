'''Classes for variable elimination Routines
   A) class Variable
      This class allows one to define Bayes Net variables.
      On initialization the variable object can be given a name and a
      domain of values. This list of domain values can be added to or
      deleted from in support of an incremental specification of the
      variable domain.
      The variable also has a set and get value method. These set a
      value for the variable that can be used by the factor class.
    B) class Factor
      This class allows one to define a factor specified by a table
      of values.
      On initialization the variables the factor is over is
      specified. This must be a list of variables. This list of
      variables cannot be changed once the constraint object is
      created.
      Once created the factor can be incrementally initialized with a
      list of values. To interact with the factor object one first
      sets the value of each variable in its scope (using the
      variable's set_value method), then one can set or get the value
      of the factor (a number) on those fixed values of the variables
      in its scope.
      Initially, one creates a factor object for every conditional
      probability table in the bayes-net. Then one initializes the
      factor by iteratively setting the values of all of the factor's
      variables and then adding the factor's numeric value using the
      add_value method.
    C) class FactorGraphClass
       This class allows one to put factors and variables together to form a Bayes net.
       It serves as a convient place to store all of the factors and variables associated
       with a Bayes Net in one place. It also has some utility routines to, e.g,., find
       all of the factors a variable is involved in.
    '''


class Variable:
    '''Class for defining Bayes Net variables. '''

    def __init__(self, name, domain=[]):
        '''Create a variable object, specifying its name (a
        string). Optionally specify the initial domain.
        The domain is the range of values we can expect a variable to take
        For example, for a binary variable this would be ['True', 'False']
        '''
        self.name = name  # text name for variable
        self.dom = list(domain)  # Make a copy of passed domain
        self.evidence_index = -1  # evidence value (stored as index into self.dom)
        self.assignment_index = -1  # For use by factors. We can assign variables values
        # and these assigned values can be used by factors
        # to index into their tables.

    def add_domain_values(self, values):
        '''Add domain values to the domain. values should be a list.'''
        for val in values: self.dom.append(val)

    def value_index(self, value):
        '''Domain values need not be numbers, so return the index
           in the domain list of a variable value'''
        return self.dom.index(value)

    def domain_size(self):
        '''Return the size of the domain'''
        return (len(self.dom))

    def domain(self):
        '''return the variable domain'''
        return (list(self.dom))

    def set_evidence(self, val):
        '''set this variable's value when it operates as evidence'''
        self.evidence_index = self.value_index(val)

    def get_evidence(self):
        return (self.dom[self.evidence_index])

    def set_assignment(self, val):
        '''Set this variable's assignment value for factor lookups'''
        self.assignment_index = self.value_index(val)

    def get_assignment(self):
        return (self.dom[self.assignment_index])

    ##These routines are special low-level routines used directly by the
    ##factor objects
    def set_assignment_index(self, index):
        '''This routine is used by the factor objects'''
        self.assignment_index = index

    def get_assignment_index(self):
        '''This routine is used by the factor objects'''
        return (self.assignment_index)

    def __repr__(self):
        '''string to return when evaluating the object'''
        return ("{}".format(self.name))

    def __str__(self):
        '''more elaborate string for printing'''
        return ("{}, Dom = {}".format(self.name, self.dom))


class Factor:
    '''Class for defining factors. A factor is a function that is over
    an ORDERED sequence of variables called its scope. It maps every
    assignment of values to these variables to a number. In a Bayes
    Net every CPT is represented as a factor. Pr(A|B,C) for example
    will be represented by a factor over the variables (A,B,C). If we
    assign A = a, B = b, and C = c, then the factor will map this
    assignment, A=a, B=b, C=c, to a number that is equal to Pr(A=a|
    B=b, C=c). During variable elimination new factors will be
    generated. However, the factors computed during variable
    elimination do not necessarily correspond to conditional
    probabilities. Nevertheless, they still map assignments of values
    to the variables in their scope to numbers.
    Note that if the factor's scope is empty it is a constaint factor
    that stores only one value. add_values would be passed something
    like [[0.25]] to set the factor's single value. The get_value
    functions will still work.  E.g., get_value([]) will return the
    factor's single value. Constaint factors might be created when a
    factor is restricted.'''

    def __init__(self, name, scope):
        '''create a Factor object, specify the Factor name (a string)
        and its scope (an ORDERED list of variable objects).'''
        self.scope = list(scope)
        self.name = name

        size = 1

        # One cell for each possible assignment of values in the scope
        for v in scope:
            size = size * v.domain_size()

        self.values = [0] * size  # initialize values to be long list of zeros.

    def get_scope(self):
        '''returns copy of scope...you can modify this copy without affecting
           the factor object'''
        return list(self.scope)

    def add_values(self, values):
        '''This routine can be used to initialize the factor. We pass
        it a list of lists. Each sublist is a ORDERED sequence of
        values, one for each variable in self.scope followed by a
        number that is the factor's value when its variables are
        assigned these values. For example, if self.scope = [A, B, C],
        and A.domain() = [1,2,3], B.domain() = ['a', 'b'], and
        C.domain() = ['heavy', 'light'], then we could pass add_values the
        following list of lists
        [[1, 'a', 'heavy', 0.25], [1, 'a', 'light', 1.90],
         [1, 'b', 'heavy', 0.50], [1, 'b', 'light', 0.80],
         [2, 'a', 'heavy', 0.75], [2, 'a', 'light', 0.45],
         [2, 'b', 'heavy', 0.99], [2, 'b', 'light', 2.25],
         [3, 'a', 'heavy', 0.90], [3, 'a', 'light', 0.111],
         [3, 'b', 'heavy', 0.01], [3, 'b', 'light', 0.1]]
         This list initializes the factor so that, e.g., its value on
         (A=2,B=b,C='light) is 2.25'''

        for t in values:
            index = 0
            for v in self.scope:
                index = index * v.domain_size() + v.value_index(t[0])
                t = t[1:]
            self.values[index] = t[0]

    def add_value_at_current_assignment(self, number):

        '''This function allows adding values to the factor in a way
        that will often be more convenient. We pass it only a single
        number. It then looks at the assigned values of the variables
        in its scope and initializes the factor to have value equal to
        number on the current assignment of its variables. Hence, to
        use this function one first must set the current values of the
        variables in its scope.
        For example, if self.scope = [A, B, C],
        and A.domain() = [1,2,3], B.domain() = ['a', 'b'], and
        C.domain() = ['heavy', 'light'], and we first set an assignment for A, B
        and C:
        A.set_assignment(1)
        B.set_assignment('a')
        C.set_assignment('heavy')
        then we call
        add_value_at_current_assignment(0.33)
         with the value 0.33, we would have initialized this factor to have
        the value 0.33 on the assigments (A=1, B='1', C='heavy')
        This has the same effect as the call
        add_values([1, 'a', 'heavy', 0.33])
        One advantage of the current_assignment interface to factor values is that
        we don't have to worry about the order of the variables in the factor's
        scope. add_values on the other hand has to be given tuples of values where
        the values must be given in the same order as the variables in the factor's
        scope.
        See recursive_print_values called by print_table to see an example of
        where the current_assignment interface to the factor values comes in handy.
        '''

        index = 0
        for v in self.scope:
            index = index * v.domain_size() + v.get_assignment_index()
        self.values[index] = number


    def get_value_at_current_assignments(self):

        '''This function is used to retrieve a value from the
        factor. The value retrieved is the value of the factor when
        evaluated at the current assignment to the variables in its
        scope.
        For example, if self.scope = [A, B, C], and A.domain() =
        [1,2,3], B.domain() = ['a', 'b'], and C.domain() = ['heavy',
        'light'], and we had previously invoked A.set_assignment(1),
        B.set_assignment('a') and C.set_assignment('heavy'), then this
        function would return the value of the factor on the
        assigments (A=1, B='1', C='heavy')'''

        index = 0
        for v in self.scope:
            index = index * v.domain_size() + v.get_assignment_index()
        return self.values[index]

    def add_value_at_assignment(self, assignment):

        '''This function allows adding values to the factor in a way
        that will often be more convenient. We pass it only a single
        number. It then looks at the assigned values of the variables
        in its scope and initializes the factor to have value equal to
        number on the current assignment of its variables. Hence, to
        use this function one first must set the current values of the
        variables in its scope.
        For example, if self.scope = [A, B, C],
        and A.domain() = [1,2,3], B.domain() = ['a', 'b'], and
        C.domain() = ['heavy', 'light'], and we first set an assignment for A, B
        and C:
        A.set_assignment(1)
        B.set_assignment('a')
        C.set_assignment('heavy')
        then we call
        add_value_at_current_assignment(0.33)
         with the value 0.33, we would have initialized this factor to have
        the value 0.33 on the assigments (A=1, B='1', C='heavy')
        This has the same effect as the call
        add_values([1, 'a', 'heavy', 0.33])
        One advantage of the current_assignment interface to factor values is that
        we don't have to worry about the order of the variables in the factor's
        scope. add_values on the other hand has to be given tuples of values where
        the values must be given in the same order as the variables in the factor's
        scope.
        See recursive_print_values called by print_table to see an example of
        where the current_assignment interface to the factor values comes in handy.
        '''

        index = 0
        for v in self.scope:
            index = index * v.domain_size() + v.value_index(assignment[0])
            assignment = assignment[1:]
        self.values[index] = assignment[0]

    def get_value(self, variable_values):

        '''This function is used to retrieve a value from the
        factor. We pass it an ordered list of values, one for every
        variable in self.scope. It then returns the factor's value on
        that set of assignments.  For example, if self.scope = [A, B,
        C], and A.domain() = [1,2,3], B.domain() = ['a', 'b'], and
        C.domain() = ['heavy', 'light'], and we invoke this function
        on the list [1, 'b', 'heavy'] we would get a return value
        equal to the value of this factor on the assignment (A=1,
        B='b', C='light')'''

        index = 0
        for v in self.scope:
            index = index * v.domain_size() + v.value_index(variable_values[0])
            variable_values = variable_values[1:]
        return self.values[index]


    def print_table(self):
        '''print the factor's table'''
        saved_values = []  # save and then restore the variable assigned values.
        for v in self.scope:
            saved_values.append(v.get_assignment_index())

        self.recursive_print_values(self.scope)

        for v in self.scope:
            v.set_assignment_index(saved_values[0])
            saved_values = saved_values[1:]

class FactorGraphClass:

    '''Class for defining a Bayes Net.
       This class is simple, it just is a wrapper for a list of factors. And it also
       keeps track of all variables in the scopes of these factors'''

    def __init__(self, name, vars, factors):
        self.name = name
        self.variables = list(vars)
        self.factors = list(factors)
        for f in self.factors:
            for v in f.get_scope():
                if not v in self.variables:
                    print("Bayes net initialization error")
                    print("Factor scope {} has variable {} that")

    def get_factors(self):
        return list(self.factors)

    def get_variables(self):
        return list(self.variables)


def rec_multiply_factors(Factors, newFactorScope, newFactor, total=1):
    '''
    :param Factors: original factors to be multiplied
    :param newFactorScope: scope after marginalization
    :param newFactor: new factor being created
    :param total: score of multiplication
    :returns: THis function basically iterates over all possible assignments of values to the cope and computes the CPT
    '''
    if (not (len(newFactorScope))):

        for factor in Factors:
            total *= factor.get_value_at_current_assignments()
        newFactor.add_value_at_current_assignment(total)

    else:
        for var_val in newFactorScope[0].domain():
            newFactorScope[0].set_assignment(var_val)
            rec_multiply_factors(Factors, newFactorScope[1:], newFactor, total)


def multiply_factors(Factors):
    '''
    @param Factors: factors to be multiplied.
    Note, each of these factors have to have their assignment prespecified i.e we are multiplying for a particular assignment of values to the factors
    @return: A new factor that is the product of the factors in Factors'''
    # Getting the new factor name
    newFactorName = ""
    for i in range(len(Factors)):
        cur_factor = Factors[i]
        if (i == 0):
            newFactorName += cur_factor.name
            continue
        if (i == len(Factors) - 1):
            newFactorName = newFactorName + "_" + cur_factor.name
            continue
        else:
            newFactorName = newFactorName + "_" + cur_factor.name + "_"
            continue

    # Getting the new factor scope
    newFactorScope = list()
    for factor in Factors:
        for var in factor.get_scope():
            if (var in newFactorScope):
                continue
            newFactorScope.append(var)

    # Creating the new factor
    newFactor = Factor(newFactorName, newFactorScope)

    # Fill up the CPT for all possible assignments of the current factor under marginalization
    rec_multiply_factors(Factors, newFactorScope, newFactor)

    return newFactor


def construct_assignment_tables(newFactor, f, newFactorScope, variable, summ=0):
    if (not (len(newFactorScope))):
        for val in variable.domain():
            variable.set_assignment(val)
            curr = f.get_value_at_current_assignments()
            summ += curr
        newFactor.add_value_at_current_assignment(summ)
    else:
        for var_val in newFactorScope[0].domain():
            newFactorScope[0].set_assignment(var_val)
            construct_assignment_tables(newFactor, f, newFactorScope[1:], variable)

def sum_out_variable(f, var):
    '''return a new factor that is the product of the factors in Factors
       followed by the suming out of Var'''

    # Set the factor name and scope
    newFactorName = f.name + "_{S: " + str(var.name) + "}"
    newFactorScope = f.get_scope()

    # Remove var from the scope of the newFactor
    newFactorScope.remove(var)

    # Create the newFactor
    newFactor = Factor(newFactorName, newFactorScope)

    # Summing
    construct_assignment_tables(newFactor, f, newFactorScope, var)
    return newFactor



def get_bfs_ordering(Factors, QueryVar):
    '''Compute a BFS ordering given a list of factors. Return a list
    of variables from the scopes of the factors in Factors. The QueryVar is
    NOT part of the returned ordering

    @returns: List of variables which represents the ordering'''
    ordering = []

    #TODO : Implement your ordering function here. Make sure to include an assert statement for the ordering you expect
    visiting = []
    visiting.append(QueryVar)

    while len(visiting) != 0:
        n = visiting.pop(0)
        ordering.append(n)
        for f in Factors:
            is_connected = False
            for v in f.get_scope():
                if v.name == n.name:
                    is_connected = True
            if is_connected:
                for v in f.get_scope():
                    is_visited = False
                    for v_visited in ordering+visiting:
                        if v_visited.name == v.name:
                            is_visited = True
                    if not is_visited:
                        visiting.append(v)
    ordering.pop(0)     # remove QueryVar
    # assertion of the correct order
    assert len(ordering) == 4
    assert ordering[0].name == "D"
    assert ordering[1].name == "B"
    assert ordering[2].name == "C"
    assert ordering[3].name == "A"
    return ordering

def _collapse(factors, elem_var, QueryVar):
    '''Remove all factors that mention a variable (elem_var) and add a new factor representing the collapsed quantity'''
    input_factor_len = len(factors)
    f_eliminated = []
    fac = list(factors)
    for f in fac:
        iselem = False
        for s in f.get_scope():
            if s.name == elem_var.name:
                iselem = True
                break
        if iselem:
            f_eliminated.append(f)
            factors.remove(f)

    # Set the factor name and scope
    newFactorName = "collapse_elemvar_" + str(elem_var.name) + "_qvar_" + str(QueryVar.name)
    newFactorScope = []
    for f in f_eliminated:
        # check for duplicated vars
        for v in f.get_scope():
            is_existed = False
            for k in newFactorScope:
                if k.name == v.name:
                    is_existed = True
            if not is_existed:
                newFactorScope.append(v)

    # Remove elem_var from the scope of the newFactor
    newFactorScope.remove(elem_var)

    # Create the newFactor
    newfactor = Factor(newFactorName, newFactorScope)

    ## construct probability table
    # construct all possible combinations of variables, stored in var_table
    var_table = []
    temp = []
    for var in newFactorScope:
        temp = list(var_table)
        var_table = []
        for dom in var.domain():
            if len(temp) == 0:
                var_table.append([dom])
            else:
                var_table.extend([x + [dom] for x in temp])

    # compute probabilities
    for v in var_table:
        p = 0.0
        for v_e in elem_var.domain():
            p_temp = 1.0
            for f in f_eliminated:
                f_input = []
                for s in f.get_scope():
                    if elem_var.name == s.name:
                        f_input.append(v_e)
                    else:
                        for idx, gv in enumerate(newFactorScope):
                            if gv.name == s.name:
                                f_input.append(v[idx])
                p_temp *= f.get_value(f_input)
            p += p_temp
        # add value to factor
        newfactor.add_values([v + [p]])
    # new factor list
    factors.append(newfactor)
    assert len(factors) == input_factor_len - len(f_eliminated) + 1
    assert newFactorScope == factors[-1].get_scope()

def compute_prob(NetFactors):

    # take product of factors with Q
    remaining_factors_w_Q = multiply_factors(NetFactors)

    # get the normalization factor
    normalization_factor = 0
    for i in remaining_factors_w_Q.values:
        normalization_factor += i

    # check if zero division will occur
    if (not (normalization_factor)):
        raise ZeroDivisionError

    # get list of posterior distribution
    list_of_pos = list()
    for i in remaining_factors_w_Q.values:
        list_of_pos.append(i / normalization_factor)

    return list_of_pos

def  Naive_VE(FactorGraph, QueryVar):
    '''
    Input: FactorGraph---a BN object (a Bayes Net)
           QueryVar---a Variable object (the variable whose distribution
                      we want to compute)
   VE returns a distribution over the values of QueryVar, i.e., a list
   of numbers one for every value in QueryVar's domain. These numbers
   sum to one, and the i'th number is the probability that QueryVar is
   equal to its i'th value.
    '''
    # TODO : Implement the naive version of variable elimination
    # tabulate store all 2^5 = 32 possibilities
    # construct all possible combinations of variables, stored in var_table
    var_table, temp = [], []
    for var in FactorGraph.get_variables():
        temp = list(var_table)
        var_table = []
        for dom in var.domain():
            if len(temp) == 0:
                var_table.append([dom])
            else:
                var_table.extend([x + [dom] for x in temp])
    # compute probabilities of the QueryVar, p
    p = [0]*QueryVar.domain_size()
    fac = FactorGraph.get_factors()
    graph_var = FactorGraph.get_variables()
    for v in var_table:
        p_temp = 1.0
        for f in fac:
            fac_input = []
            for s in f.get_scope():
                for idx, gv in enumerate(graph_var):
                    if gv.name == s.name:
                        fac_input.append(v[idx])
            p_temp *= f.get_value(fac_input)
        idx = None
        for i, value in enumerate(graph_var):
            if value.name == QueryVar.name:
                idx = i
                break
        if not idx is None:
            p[QueryVar.domain().index(v[idx])] += p_temp
    return p

def __variable_elimination(FactorGraph, QueryVar):
    '''
    Input: FactorGraph---a BN object (a Bayes Net)
           QueryVar---a Variable object (the variable whose distribution
                      we want to compute)
   VE returns a distribution over the values of QueryVar, i.e., a list
   of numbers one for every value in QueryVar's domain. These numbers
   sum to one, and the i'th number is the probability that QueryVar is
   equal to its i'th value.
    '''
    factors = FactorGraph.get_factors()
    for each_var in get_bfs_ordering(factors, QueryVar):
        _collapse(factors, each_var, QueryVar)

    return compute_prob(factors)

def __initialize(name):
    '''Bayes net for figure shown in handout'''

    A = Variable("A", ["True", "False"])
    B = Variable("B", ["True", "False"])
    C = Variable("C", ["True", "False"])
    D = Variable("D", ["True", "False"])
    E = Variable("E", ["True", "False"])
    vars = [A, B, C, D, E]

    psi_a = Factor("psi_a", [A])
    psi_b = Factor("psi_b", [B, A])
    psi_c = Factor("psi_c", [C, A])
    psi_d = Factor("psi_d", [D, B, C])
    psi_e = Factor("psi_e", [E, D])

    psi_a.add_values([["True", 0.6], ["False", 0.4]])
    psi_b.add_values(
        [["True", "True", 0.2], ["False", "True", 0.8], ["True", "False", 0.75], ["False", "False", 0.25]])
    psi_c.add_values(
        [["True", "True", 0.8], ["False", "True", 0.2], ["True", "False", 0.1], ["False", "False", 0.9]])
    psi_d.add_values(
        [["True", "True", "True", 0.95], ["False", "True", "True", 0.05], ["True", "True", "False", 0.9],
         ["False", "True", "False", 0.1], ["True", "False", "True", 0.8], ["False", "False", "True", 0.2],
         ["True", "False", "False", 0.0], ["False", "False", "False", 1.0]])
    psi_e.add_values(
        [["True", "True", 0.7], ["False", "True", 0.3], ["True", "False", 0.0], ["False", "False", 1.0]])
    factors = [psi_a, psi_b, psi_c, psi_d, psi_e]

    FactorGraph = FactorGraphClass(name, vars, factors)
    return FactorGraph, E

if __name__ == "__main__":
    name = "My Factor Graph"
    FactorGraph, E = __initialize(name)
    prob_dist = __variable_elimination(FactorGraph, E)
    print(prob_dist)
    prob_dist =  Naive_VE(FactorGraph, E)
    print(prob_dist)

