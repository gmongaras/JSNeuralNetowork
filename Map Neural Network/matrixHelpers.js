// Code by John Remmen
// Code found at: https://gist.github.com/jremmen/9454479
// Editted by Florian Maul

mmultiply = (a, b) => a.map(x => transpose(b).map(y => dotproduct(x, y)));
dotproduct = (a, b) => a.map((x, i) => a[i] * b[i]).reduce((m, n) => m + n);
transpose = a => a[0].map((x, i) => a.map(y => y[i]));



// Code created by me
ssum = (a, b) => a.map((x, i) => a[i] + b[i]);