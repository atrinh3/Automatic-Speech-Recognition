class HiddenMarkovModel:	
    def __init__(self, name):
        self.name = name
		self.mean = [None]
		self.covariance = [None]
		self.transition = [None]
		self.initial = [None]
	
	def update_mean(self, mean_matrix):
	    self.mean = mean_matrix
	
	def update_covariance(self, covariance_matrix):
	    self.covariance = covariance_matrix
	
	def update_transition(self, transition_matrix):
	    self.transition = transition_matrix
	
	def update_initial(self, initial_state)
	    self.initial = initial_state
		
    def get_mean():
	    return self.mean
		
	def get_covariance():
	    return self.covariance
		
	def get_transition():
	    return self.transition
		
	def get_initial():
	    return self.initial
