package main

import (
	"bufio"
	"flag"
	"fmt"
	"image/color"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

type datum struct {
	x1         float64
	x2         float64
	label      int
	prediction int
	distance   float64
}

func main() {
	var inFlag string
	flag.StringVar(&inFlag, "i", "base1.txt", "data input file")
	var ratioFlag string
	flag.StringVar(&ratioFlag, "r", "0.8", "ratio for the train/test split")
	var k int
	flag.IntVar(&k, "k", 4, "number of neighbors to consider")

	flag.Parse()

	ratio, err := strconv.ParseFloat(ratioFlag, 64)
	if err != nil {
		fmt.Fprintf(os.Stderr, "could not parse ratio %v : %v", ratioFlag, err)
	}

	dataFile, err := os.Open(inFlag)
	if err != nil {
		fmt.Fprintf(os.Stderr, "could not open file %v for reading : %v", inFlag, err)
	}

	fmt.Printf("Input file : %v, train/test ratio : %v, k=%v\n", inFlag, ratio, k)

	var data = []datum{}
	scanner := bufio.NewScanner(dataFile)
	for scanner.Scan() {
		row := scanner.Text()
		data = append(data, newDatum(row))
	}

	shuffle(data)
	trainData, testData := splitData(data, ratio)

	fmt.Printf("Size of data : \n")
	fmt.Printf("\t - train : %v\n", len(trainData))
	fmt.Printf("\t - test : %v\n", len(testData))

	for key, v := range testData {
		neighbors := v.getNeighbors(trainData, k)
		fmt.Printf("%v\n", neighbors)
		testData[key].prediction = v.getClass(neighbors)
	}
	fmt.Printf("Test data : \n%v", testData)
	accuracy, errors := getAccuracy(testData)
	fmt.Printf("Classification accuracy is %v\n", accuracy)
	fmt.Printf("Missclassified elements : %v\n%v\n", len(errors), errors)

	p, err := plot.New()
	if err != nil {
		fmt.Fprintf(os.Stderr, "could not create new plot : %v", err)
		os.Exit(1)
	}

	p.Title.Text = "Scatter plot of data"
	p.X.Label.Text = "x1"
	p.Y.Label.Text = "x2"

	trainScatterData := createScatterData(trainData, false)
	trainScatter, err := plotter.NewScatter(trainScatterData)
	if err != nil {
		fmt.Fprintf(os.Stderr, "could not create new scatter plot : %v", err)
		os.Exit(1)
	}
	trainScatter.GlyphStyleFunc = scatterStyle(trainScatterData, false)
	p.Add(trainScatter)

	testScatterData := createScatterData(testData, true)
	testScatter, err := plotter.NewScatter(testScatterData)
	if err != nil {
		fmt.Fprintf(os.Stderr, "could not create new scatter plot : %v", err)
		os.Exit(1)
	}
	testScatter.GlyphStyleFunc = scatterStyle(testScatterData, true)
	p.Add(testScatter)

	if err = p.Save(800, 800, "scatter.png"); err != nil {
		fmt.Fprintf(os.Stderr, "could not save plot : %v", err)
		os.Exit(2)
	}
}

func newDatum(row string) datum {
	r := strings.Split(row, " ")
	x1, err := strconv.ParseFloat(r[0], 64)
	if err != nil {
		fmt.Fprintf(os.Stderr, "could not parse float %v : %v", x1, err)
	}
	x2, err := strconv.ParseFloat(r[1], 64)
	if err != nil {
		fmt.Fprintf(os.Stderr, "could not parse float %v : %v", x2, err)
	}
	label, err := strconv.Atoi(r[2])
	if err != nil {
		fmt.Fprintf(os.Stderr, "could not parse int %v : %v", label, err)
	}
	return datum{x1: x1, x2: x2, label: label}
}

func (d datum) String() string {
	return fmt.Sprintf("distance : %v, x1: %v, x2: %v, label : %v, predicted : %v\n", d.distance, d.x1, d.x2, d.label, d.prediction)
}

func (d *datum) euclideanDistance(testInstance *datum) {
	d.distance = math.Sqrt(math.Pow(d.x1-testInstance.x1, 2) + math.Pow(d.x2-testInstance.x2, 2))
}

func (d *datum) getNeighbors(train []datum, k int) []datum {
	distances := make([]datum, len(train))
	for i, v := range train {
		v.euclideanDistance(d)
		distances[i] = v
	}
	sort.SliceStable(distances, func(i, j int) bool { return distances[i].distance < distances[j].distance })
	return distances[len(distances)-k:]
}

func (d *datum) getClass(neighbors []datum) int {
	votes := make(map[int]int)
	for _, v := range neighbors {
		if _, ok := votes[v.label]; ok {
			votes[v.label]++
		} else {
			votes[v.label] = 1
		}
	}
	var max int
	var class int
	for key, value := range votes {
		if value > max {
			max = value
			class = key
		}
	}
	return class
}

func shuffle(arr []datum) {
	rand.Seed(1337)
	//rand.Seed(time.Now().UTC().UnixNano())
	for i := len(arr) - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		arr[i], arr[j] = arr[j], arr[i]
	}
}

func splitData(arr []datum, ratio float64) ([]datum, []datum) {
	splitIndex := int(ratio * float64(len(arr)))
	return arr[:splitIndex], arr[splitIndex:]
}

func getAccuracy(arr []datum) (float64, []datum) {
	var ok int
	errors := []datum{}
	for _, v := range arr {
		if v.label == v.prediction {
			ok++
		} else {
			errors = append(errors, v)
		}

	}
	return float64(ok) / float64(len(arr)), errors
}

func createScatterData(arr []datum, test bool) plotter.XYZs {
	data := make(plotter.XYZs, len(arr))
	for i := range arr {
		data[i].X, data[i].Y = arr[i].x1, arr[i].x2
		if test {
			data[i].Z = float64(arr[i].prediction)
		} else {
			data[i].Z = float64(arr[i].label)
		}
	}
	return data
}

func scatterStyle(data plotter.XYZs, test bool) func(int) draw.GlyphStyle {
	return func(i int) draw.GlyphStyle {
		_, _, z := data.XYZ(i)
		var c color.Color
		var s draw.GlyphDrawer
		if test {
			s = draw.PlusGlyph{}
		} else {
			s = draw.PyramidGlyph{}
		}
		switch int(z) {
		case 1:
			c = color.RGBA{R: 255, A: 255}
		case 2:
			c = color.RGBA{G: 255, A: 255}
		case 3:
			c = color.RGBA{B: 255, A: 255}
		}
		return draw.GlyphStyle{Color: c, Radius: vg.Points(3), Shape: s}
	}
}
