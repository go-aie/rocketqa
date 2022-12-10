package internal_test

import (
	"testing"

	"github.com/RussellLuo/go-rocketqa/internal"
	"github.com/google/go-cmp/cmp"
)

func TestMatrix_Norm(t *testing.T) {
	tests := []struct {
		inTensor internal.Tensor
		wantData []float32
	}{
		{
			inTensor: internal.Tensor{
				Shape: []int32{5, 1},
				Data:  []float32{1, 2, 3, 4, 5},
			},
			wantData: []float32{0.13483997, 0.26967994, 0.40451991, 0.53935989, 0.67419986},
		},
	}
	for _, tt := range tests {
		m := internal.NewMatrix(tt.inTensor)
		gotData := m.Norm().RawData()
		if !cmp.Equal(gotData, tt.wantData) {
			diff := cmp.Diff(gotData, tt.wantData)
			t.Errorf("Want - Got: %s", diff)
		}
	}
}
