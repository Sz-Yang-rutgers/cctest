#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <string>

//constexpr double G = 5.67E-11; // universal gravitational constant

class Body {
private:
	std::string name;
	const double m; // rest mass
	double x,y,z; // location
	double vx, vy, vz; // velocity
public:
	Body(std::string name, double m, double x, double y, double z,
			 double vx, double vy, double vz) :
		name(name), m(m), x(x), y(y), z(z), vx(vx), vy(vy), vz(vz) {}
	
	double get_x() const{ return this->x; }
	double get_y() const{ return this->y; }
	double get_z() const{ return this->z; }
	double get_vx() const{ return this->vx; }
	double get_vy() const{ return this->vy; }
	double get_vz() const{ return this->vz; }
    double get_m()  const{ return this->m; }

    void set_x(double x) { this->x = x; }
    void set_y(double y) { this->y = y; }
    void set_z(double z) { this->z = z; }
    void set_vx(double vx) { this->vx = vx; }
    void set_vy(double vy) { this->vy = vy; }
    void set_vz(double vz) { this->vz = vz; }   

    std::string get_name() const { return name; }
	// stream operator declared as a friend and defined outside the class
	friend std::ostream& operator<<(std::ostream& s, const Body& b);
	// trivial destructor
	~Body() {}

double dist(Body b) {
	double dx = this->x - b.get_x(); 
	double dy = this->y - b.get_y(); 
	double dz = this->z - b.get_z(); 
	return sqrt(dx * dx + dy * dy + dz * dz);
}

double d2x(Body b, double a) {
	double dx = this->get_x() - b.get_x();
	return a * dx / this->dist(b);
}

double d2y(Body b, double a) {
	double dy = this->y - b.get_y();
	return a * dy / this->dist(b);
}

double d2z(Body b, double a) {
	double dz = this->z - b.get_z();
	return a * dz / this->dist(b);
}

// step this one body forward in time
void step_forward(double dt) {
	this->set_x( this->get_x() + this->get_vx() * dt);
	this->set_y( this->get_y() + this->get_vy() * dt);
	this->set_z( this->get_z() + this->get_vz() * dt);
}


};

// stream output operator defined outside the class
inline std::ostream& operator<<(std::ostream& s, const Body& b) {
	return s << std::setw(14) << b.name << std::setw(14) << b.x << std::setw(14) << b.y <<std::setw(14)<<b.z;
}
