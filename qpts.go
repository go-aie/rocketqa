package rocketqa

// QPT is a structured type that represents a single parameter group.
type QPT struct {
	Query string
	Para  string
	Title string
}

// QPTs is a helper type that makes it easy to construct parameters in a more structured way.
type QPTs []QPT

// Q collects the values of the Query fields from all elements and return them as a slice.
func (qs QPTs) Q() (queries []string) {
	for _, q := range qs {
		queries = append(queries, q.Query)
	}
	return
}

// P collects the values of the Para fields from all elements and return them as a slice.
func (qs QPTs) P() (paras []string) {
	for _, q := range qs {
		paras = append(paras, q.Para)
	}
	return
}

// T collects the values of the Title fields from all elements and return them as a slice.
func (qs QPTs) T() (titles []string) {
	for _, q := range qs {
		titles = append(titles, q.Title)
	}
	return
}
