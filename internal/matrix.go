package internal

import (
	"gonum.org/v1/gonum/mat"
)

type Matrix struct {
	m *mat.Dense
}

func NewMatrix(t Tensor) *Matrix {
	if len(t.Shape) != 2 {
		panic("t is not a matrix")
	}

	data, ok := t.Data.([]float32)
	if !ok {
		panic("t.Data is not of type []float32")
	}

	m := mat.NewDense(int(t.Shape[0]), int(t.Shape[1]), Float32To64(data))
	return &Matrix{m: m}
}

func (m *Matrix) Row(i int) []float32 {
	r := mat.Row(nil, i, m.m)
	return Float64To32(r)
}

func (m *Matrix) Col(j int) []float32 {
	c := mat.Col(nil, j, m.m)
	return Float64To32(c)
}

func (m *Matrix) Rows() [][]float32 {
	var rows [][]float32

	r, _ := m.m.Dims()
	for i := 0; i < r; i++ {
		rows = append(rows, m.Row(i))
	}

	return rows
}

func (m *Matrix) Cols() [][]float32 {
	var cols [][]float32

	_, c := m.m.Dims()
	for j := 0; j < c; j++ {
		cols = append(cols, m.Col(j))
	}

	return cols
}

func (m *Matrix) Norm() *Matrix {
	r, c := m.m.Dims()

	norm := m.m.Norm(2) // the square root of the sum of the squares of the elements
	data := make([]float64, r*c)
	for i := 0; i < len(data); i++ {
		data[i] = norm
	}

	normDense := mat.NewDense(r, c, data)
	m.m.DivElem(m.m, normDense)

	return m
}

func (m *Matrix) RawData() []float32 {
	return Float64To32(m.m.RawMatrix().Data)
}

func Float32To64(v []float32) []float64 {
	var result []float64
	for _, value := range v {
		result = append(result, float64(value))
	}
	return result
}

func Float64To32(v []float64) []float32 {
	var result []float32
	for _, value := range v {
		result = append(result, float32(value))
	}
	return result
}
