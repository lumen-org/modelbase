

def _Modeldata_field():
   """Returns a new field that represents the imaginary dimension of 'model vs data'."""
   return Field('model vs data', dm.DiscreteDomain(), dm.DiscreteDomain(['model', 'data']), dtype='string')

"""
In fact unified queries are possible, by means of an additional 'artificial' field 'model vs data'. That is an
ordinal field with the two values "model" and "data". It can only be filtered or split by this field, and the
results are accordingly for the results from the data or the model, respectively.

Internal:
    Internally queries against the model and its data are answered differently. Also that field does not actually
    exists in the model. Queries may contain it, however they are translated onto the 'internal' interface in the
    .marginalize, .condition, .predict-methods.
"""


"""
Args:
    is_pure: An performance optimization parameter. Set to True if you can guarantee that keep and remove do not contain the special value "model vs data".
"""


def marginalize():
    # Data marginalization
    if self.mode == 'both' or self.mode == 'data':
        # Note: we never need to copy data, since we never change data. creating views is enough
        self.data = self.data.loc[:, keep]
        self.test_data = self.test_data.loc[:, keep]
        if self.mode == 'data':
            self._update_remove_fields(remove)
            return self


"""
model.density():
        Special field 'model vs data':

            * you must not use option (1) # TODO?
            * if you use option (2): supply it as the last element of the domain list
            * if you use option (3): use the key 'model vs data' for its domain

        You may pass a special field with name 'model vs data':
          * If its value is 'data' then it is interpreted as a density query against the data, i.e. frequency of items is returned.
          * If the value is 'model' it is interpreted as a query against the model, i.e. density of input is returned.
          * If value is 'both' a 2-tuple of both is returned.
          
    # data frequency
    if mode == "both" or mode == "data":
        # TODO?? should I do this against the test data as well?? expand mode to test data or something? also applies to similar sections in model.probability
        # cnt = len(data_ops.condition_data(self.data, zip(self.names, ['=='] * self.dim, values)))
        cnt = data_ops.density(self.data, values)
        if mode == "data":
            return cnt
            
    # data probability
        if mode == "both" or mode == "data":
            data_prob = data_ops.probability(self.data, domains)
            if mode == "data":
                return data_prob            
"""



def predict():
    # How to handle the 'artificial field' 'model vs data':
    # The possible cases, as which 'model vs data' occur are:
    #   * as a filter: that sets the internal mode of the model. Is handled in '.condition'
    #   * as a split: splits by the values of that field. Is handled where we handle splits.
    #   * not at all: defaults to a filter to equal 'model' and a default identity split on it, ONLY if
    #       self.mode is 'both'
    #   * in predict-clause or not -> need to assemble result frame correctly
    #   * in aggregation: raise error

    # set default filter and split on 'model vs data' if necessary
    if 'model vs data' not in split_names \
            and 'model vs data' not in filter_names \
            and self.mode == 'both':
        where.append(Condition('model vs data', '==', 'model'))
        filter_names.append('model vs data')
        splitby.append(Split('model vs data', 'identity', []))
        split_names.append('model vs data')