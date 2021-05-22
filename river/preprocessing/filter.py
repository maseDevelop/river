import typing
import operator as o
from river import base




class FixedThresholdFilter(base.Transformer):

    def __init__(self,features: typing.Tuple[base.typing.FeatureName],operator,threshold,logical_operator):
        self.features = features
        self.operator = operator
        self.threshold = threshold
        self.logical_operator = logical_operator

        operator_dict ={
            "<":o.lt,
            "<=":o.le,
            "==":o.eq,
            ">=":o.ge,
            ">":o.gt,
        }

        self.is_or = True if logical_operator == "OR" else False
        self.comparison_operator = operator_dict[self.operator]









    def learn_one(self, x: dict, **kwargs) -> "Transformer":
        """Update with a set of features `x`.

        A lot of transformers don't actually have to do anything during the `learn_one` step
        because they are stateless. For this reason the default behavior of this function is to do
        nothing. Transformers that however do something during the `learn_one` can override this
        method.

        Parameters
        ----------
        x
            A dictionary of features.
        kwargs
            Some models might allow/require providing extra parameters, such as sample weights.

        Returns
        -------
        self

        """
        return self

    def transform_one(self, x: dict) -> dict:
        """Transform a set of features `x`.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
        The transformed values.

        """

        if self.is_or:
            for f in self.features:
                if(self.comparison_operator(x[f],self.threshold)):
                    #Remove instance
                    return "drop instance"
            return x

        else:
            for f in self.features:
                if(not self.comparison_operator(x[f],self.threshold)):
                    return x
            #Remove instances
            return "drop instance"
